/**************************************************************************/
/*  physics_server_3d_wrap_mt.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "physics_server_3d_wrap_mt.h"

#include "core/config/engine.h"
#include "core/os/os.h"

#include <thread>

// A wait on the clock, not on the OS sleep. OS::delay_usec is Sleep(1) on Windows for anything
// under a millisecond: a millisecond or two at best, 15 ms when the process loses its timer
// resolution, and a kilohertz loop cannot pay that per slice (the first runs ticked at a
// fraction of the rate and every park cost the main thread tens of milliseconds). Long waits
// take one OS sleep with a margin; the rest is spent on the clock, yielding the core between
// reads so a ready thread can have it.
static void _free_wait_usec(uint64_t p_usec) {
	OS *os = OS::get_singleton();
	const uint64_t target = os->get_ticks_usec() + p_usec;
	if (p_usec > 2500) {
		os->delay_usec((uint32_t)(p_usec - 1500));
	}
	while (os->get_ticks_usec() < target) {
		std::this_thread::yield();
	}
}

void PhysicsServer3DWrapMT::_assign_mt_ids(WorkerThreadPool::TaskID p_pump_task_id) {
	server_thread = Thread::get_caller_id();
	server_task_id = p_pump_task_id;
}

void PhysicsServer3DWrapMT::_thread_exit() {
	exit = true;
}

void PhysicsServer3DWrapMT::_thread_loop() {
	while (!exit) {
		WorkerThreadPool::get_singleton()->yield();

		if (!doing_sync.is_set()) {
			command_queue.flush_all();
		}
	}
}

void PhysicsServer3DWrapMT::_thread_sync() {
	doing_sync.set();
}

/* CRUMB FREE-RUNNING LOOP */

void PhysicsServer3DWrapMT::_free_thread_entry(void *p_self) {
	static_cast<PhysicsServer3DWrapMT *>(p_self)->_free_thread_loop();
}

void PhysicsServer3DWrapMT::_free_pause_delta(int p_delta) {
	free_pause_depth = MAX(0, free_pause_depth + p_delta);
}

void PhysicsServer3DWrapMT::_free_set_callback(const Callable &p_callback) {
	MutexLock lock(free_callback_mutex);
	free_callback = p_callback;
}

void PhysicsServer3DWrapMT::set_free_running_callback(const Callable &p_callback) {
	_free_set_callback(p_callback);
}

void PhysicsServer3DWrapMT::set_free_running_paused(bool p_paused) {
	if (!free_running) {
		return;
	}
	if (Thread::get_caller_id() == server_thread) {
		_free_pause_delta(p_paused ? 1 : -1);
		return;
	}
	if (!p_paused) {
		// The release is asynchronous; the parked loop picks it up within a slice.
		command_queue.push(this, &PhysicsServer3DWrapMT::_free_pause_delta, -1);
		return;
	}
	// A park must not return while a tick is in flight. The loop drains this queue inside every
	// server call the tick makes (the wrapper's server-thread branch flushes pending commands), so
	// a synchronous push alone can return from the middle of a tick with the step still to come,
	// and the caller then reads and writes bodies the step is moving. Wait instead for the loop to
	// ACKNOWLEDGE the park from the top of an iteration: an acknowledgement counted after the push
	// means the loop is between ticks with the depth above zero, and the depth cannot return to
	// zero until this thread releases it.
	const uint64_t before = free_park_acks.load();
	command_queue.push(this, &PhysicsServer3DWrapMT::_free_pause_delta, 1);
	while (free_park_acks.load() == before && free_thread.is_started() && !free_exit.is_set()) {
		std::this_thread::yield();
	}
}

void PhysicsServer3DWrapMT::_free_thread_loop() {
	Thread::set_name("PhysicsServer3D free-running");
	OS *os = OS::get_singleton();
	Engine *engine = Engine::get_singleton();
	uint64_t next_tick_usec = os->get_ticks_usec();
	uint64_t last_step_start_usec = next_tick_usec;
	bool parked_sync = false;   // Jolt's doing_sync is held while parked so main-thread state reads pass its guard

	while (!free_exit.is_set()) {
		// The main thread's commands (bodies, areas, joints, the park itself) execute here.
		command_queue.flush_all();

		if (free_pause_depth > 0 || !free_active.load()) {
			if (!parked_sync) {
				physics_server_3d->sync();
				parked_sync = true;
			}
			free_parked.store(true);
			free_park_acks.fetch_add(1);
			// Idle, the parked wait is an OS sleep too: a command waits a millisecond or two, and
			// the core is free.
			if (free_idle.load()) {
				os->delay_usec(1000);
			} else {
				_free_wait_usec(50);
			}
			// The clock keeps running through a park. A short one (the main loop's per-frame
			// window, a couple of milliseconds) is caught up at the cap below, so simulated time
			// tracks the wall clock over a second; a long one (a bake, a sweep) is past the cap
			// and its time is dropped rather than burst through.
			continue;
		}
		if (parked_sync) {
			physics_server_3d->end_sync();
			parked_sync = false;
		}
		free_parked.store(false);

		// Idle: a token rate on OS sleeps. The world keeps stepping (pending bodies enter the
		// broadphase, monitors and contacts stay current) at no cost; the tick rate and the
		// unthrottled pacing return the moment the application lifts it.
		const bool idle = free_idle.load();
		const int tps = idle ? FREE_IDLE_TPS : MAX(1, engine->get_physics_ticks_per_second());
		const double period = 1.0 / tps;
		const uint64_t period_usec = MAX((uint64_t)1, (uint64_t)(period * 1000000.0));
		uint64_t now = os->get_ticks_usec();
		double delta;

		if (free_unthrottled && !idle) {
			// Back to back, each step advancing the wall time not yet simulated: never more than
			// one period (the configured rate is the floor of the resolution, and a slow step
			// cannot feed itself a growing delta), never less than a twentieth of one. Time the
			// thread could not step, a park or a stall, is owed and paid back in period-sized steps
			// up to the catch-up cap, so simulated time tracks the wall clock, which the coupling
			// to the electrical thread needs; beyond the cap the time is dropped.
			free_sim_debt_usec += now - last_step_start_usec;
			const uint64_t max_debt_usec = period_usec * (uint64_t)free_max_catchup;
			if (free_sim_debt_usec > max_debt_usec) {
				free_sim_debt_usec = max_debt_usec;
			}
			const uint64_t step_usec = MIN(free_sim_debt_usec, period_usec);
			free_sim_debt_usec -= step_usec;
			delta = MAX((double)step_usec / 1000000.0, period * 0.05);
		} else {
			if (now < next_tick_usec) {
				// Wait in short slices so a command from the main thread never waits long; idle,
				// each slice is an OS sleep.
				uint64_t remaining = next_tick_usec - now;
				if (idle) {
					os->delay_usec((uint32_t)MIN(remaining, (uint64_t)1000));
				} else {
					_free_wait_usec(MIN(remaining, (uint64_t)200));
				}
				continue;
			}
			// An idle step advances one NORMAL period of simulated time, not the idle period: the
			// world is meant to be standing still, and a dynamic body that exists anyway (the
			// moment between a bake and the application lifting idle) must not leap a twentieth
			// of a second in one step and bounce.
			delta = idle ? MIN(period, 1.0 / MAX(1, engine->get_physics_ticks_per_second())) : period;
			next_tick_usec += period_usec;
			if (now > next_tick_usec && (now - next_tick_usec) > period_usec * (uint64_t)free_max_catchup) {
				next_tick_usec = now;   // too far behind: drop the time rather than burst through it
			}
		}
		last_step_start_usec = now;
		delta *= engine->get_effective_time_scale();

		// The tick. Jolt's doing_sync is held across the callback so its state getters accept
		// this thread (the same window the main thread's physics callbacks had).
		physics_server_3d->sync();
		Callable cb;
		{
			MutexLock lock(free_callback_mutex);
			cb = free_callback;
		}
		if (cb.is_valid()) {
			Variant delta_variant = delta;
			const Variant *args[1] = { &delta_variant };
			Callable::CallError ce;
			Variant ret;
			cb.callp(args, 1, ret, ce);
			if (unlikely(ce.error != Callable::CallError::CALL_OK)) {
				ERR_PRINT_ONCE("PhysicsServer3D free-running callback failed: " + Variant::get_callable_error_text(cb, args, 1, ce));
			}
		}
		physics_server_3d->end_sync();

		const uint64_t step_begin = os->get_ticks_usec();
		physics_server_3d->step((real_t)delta);
		const uint64_t step_end = os->get_ticks_usec();
		free_last_step_usec.store(step_end - step_begin);
		free_last_tick_usec.store(step_end - now);
		free_tick_count.fetch_add(1);
	}
}

/* EVENT QUEUING */

void PhysicsServer3DWrapMT::step(real_t p_step) {
	if (free_running) {
		return;   // the loop steps on its own clock
	}
	if (create_thread) {
		command_queue.push(physics_server_3d, &PhysicsServer3D::step, p_step);
	} else {
		physics_server_3d->step(p_step);
	}
}

void PhysicsServer3DWrapMT::sync() {
	if (free_running) {
		// The main loop's physics window: from here to end_sync the loop is parked between ticks,
		// so the query callbacks and every node's physics callback see a world that is not
		// stepping, the same window they had when the main thread ran the step itself.
		if (!free_synced) {
			set_free_running_paused(true);
			free_synced = true;
		}
		return;
	}
	if (create_thread) {
		command_queue.push_and_sync(this, &PhysicsServer3DWrapMT::_thread_sync);
	} else {
		command_queue.flush_all(); // Flush all pending from other threads.
	}
	physics_server_3d->sync();
}

void PhysicsServer3DWrapMT::flush_queries() {
	if (free_running) {
		// Only inside the sync window: the query callbacks (RigidBody3D state sync, Area3D
		// monitors) walk the spaces' callback lists, which the step appends to.
		if (free_synced) {
			physics_server_3d->flush_queries();
		}
		return;
	}
	physics_server_3d->flush_queries();
}

void PhysicsServer3DWrapMT::end_sync() {
	if (free_running) {
		if (free_synced) {
			free_synced = false;
			set_free_running_paused(false);
		}
		return;
	}
	physics_server_3d->end_sync();

	if (create_thread) {
		doing_sync.clear();
	}
}

void PhysicsServer3DWrapMT::init() {
	if (free_running) {
		// A dedicated thread, not the pool's pump task: the loop must wake on its own clock, and a
		// pool task can only yield until a command arrives. Commands are served by the loop's own
		// flush; the sync waits in the queue are satisfied by that flush like any other.
		free_thread.start(&PhysicsServer3DWrapMT::_free_thread_entry, this);
		command_queue.push(this, &PhysicsServer3DWrapMT::_assign_mt_ids, WorkerThreadPool::INVALID_TASK_ID);
		command_queue.push_and_sync(physics_server_3d, &PhysicsServer3D::init);
	} else if (create_thread) {
		WorkerThreadPool::TaskID tid = WorkerThreadPool::get_singleton()->add_task(callable_mp(this, &PhysicsServer3DWrapMT::_thread_loop), true, "Physics server 3D pump task", true);
		command_queue.set_pump_task_id(tid);
		command_queue.push(this, &PhysicsServer3DWrapMT::_assign_mt_ids, tid);
		command_queue.push_and_sync(physics_server_3d, &PhysicsServer3D::init);
		DEV_ASSERT(server_task_id == tid);
	} else {
		server_thread = Thread::MAIN_ID;
		physics_server_3d->init();
	}
}

void PhysicsServer3DWrapMT::finish() {
	if (free_running) {
		command_queue.push_and_sync(physics_server_3d, &PhysicsServer3D::finish);
		free_exit.set();
		if (free_thread.is_started()) {
			free_thread.wait_to_finish();
		}
		server_thread = Thread::MAIN_ID;
	} else if (create_thread) {
		command_queue.push(physics_server_3d, &PhysicsServer3D::finish);
		command_queue.push(this, &PhysicsServer3DWrapMT::_thread_exit);
		if (server_task_id != WorkerThreadPool::INVALID_TASK_ID) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(server_task_id);
			server_task_id = WorkerThreadPool::INVALID_TASK_ID;
		}
		server_thread = Thread::MAIN_ID;
	} else {
		physics_server_3d->finish();
	}
}

PhysicsServer3DWrapMT::PhysicsServer3DWrapMT(PhysicsServer3D *p_contained, bool p_create_thread) {
	physics_server_3d = p_contained;
	create_thread = p_create_thread;
#ifdef THREADS_ENABLED
	free_running = create_thread && (bool)GLOBAL_GET("physics/3d/free_running");
	free_unthrottled = (bool)GLOBAL_GET("physics/3d/free_running_unthrottled");
	free_max_catchup = MAX(1, (int)GLOBAL_GET("physics/3d/free_running_max_catchup_steps"));
#endif
}

PhysicsServer3DWrapMT::~PhysicsServer3DWrapMT() {
	memdelete(physics_server_3d);
}
