/**************************************************************************/
/*  jolt_space_3d.h                                                       */
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

#pragma once

#include "jolt_soft_patch_3d.h"

#include "core/os/mutex.h"
#include "core/templates/local_vector.h"
#include "core/templates/safe_refcount.h"
#include "servers/physics_3d/physics_server_3d.h"

#include "Jolt/Jolt.h"

#include "Jolt/Core/JobSystem.h"
#include "Jolt/Core/TempAllocator.h"
#include "Jolt/Physics/Body/BodyInterface.h"
#include "Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h"
#include "Jolt/Physics/Collision/NarrowPhaseQuery.h"
#include "Jolt/Physics/Constraints/Constraint.h"
#include "Jolt/Physics/PhysicsSystem.h"

class JoltArea3D;
class JoltBody3D;
class JoltBodyActivationListener3D;
class JoltContactListener3D;
class JoltJoint3D;
class JoltLayers;
class JoltObject3D;
class JoltPhysicsDirectSpaceState3D;
class JoltShapedObject3D;
class JoltSoftBody3D;

class JoltSpace3D {
	Mutex pending_objects_mutex;
	Mutex body_call_queries_mutex;

	SelfList<JoltBody3D>::List body_call_queries_list;
	SelfList<JoltArea3D>::List area_call_queries_list;
	SelfList<JoltShapedObject3D>::List shapes_changed_list;
	SelfList<JoltShapedObject3D>::List needs_optimization_list;

	LocalVector<JPH::BodyID> pending_objects_sleeping;
	LocalVector<JPH::BodyID> pending_objects_awake;

	RID rid;

	JPH::JobSystem *job_system = nullptr;
	JPH::TempAllocator *temp_allocator = nullptr;
	JoltLayers *layers = nullptr;
	JoltContactListener3D *contact_listener = nullptr;
	JoltBodyActivationListener3D *body_activation_listener = nullptr;
	JPH::PhysicsSystem *physics_system = nullptr;
	JoltPhysicsDirectSpaceState3D *direct_state = nullptr;
	JoltArea3D *default_area = nullptr;

	float last_step = 0.0f;

	// CRUMB: the serial of this space's last step, from one counter shared by every space, so no two
	// steps of any spaces share one. A body's solved-contact slot written in another step reads zero.
	uint64_t step_serial = 0;

	bool active = false;
	bool stepping = false;

public:
	// CRUMB: what the contact listener did with soft contacts this step (the status line and the probes)
	enum SoftCount {
		SOFT_COUNT_BUILT, // patches the listener built
		SOFT_COUNT_OUTSIDE_TABLE, // curved patches outside the layer table (the split law on the grid)
		SOFT_COUNT_NOT_SETTLED, // face patches whose contact set did not settle within the pass cap
		SOFT_COUNT_SWEEP_PATH, // calls from the sweep path on soft pairs (no patch, the material's restitution)
		SOFT_COUNT_BAKED_RIGID, // soft contacts kept rigid: threaded simulation or velocity steps below 8
		SOFT_COUNT_NO_TAIL, // soft contacts kept rigid: the manifold reserved no tail
		SOFT_COUNT_SOLID_FACE, // Solid face pairs, rigid by design
		SOFT_COUNT_UNCLASSED, // pairs the builder could not class (or whose footprints miss): rigid
		SOFT_COUNT_MAX,
	};

private:
	// CRUMB: the soft contact's per-step state
	SafeNumeric<uint32_t> soft_counts[SOFT_COUNT_MAX];
	uint32_t soft_counts_last[SOFT_COUNT_MAX] = {};
	JPH::ContactConstraintManager::SoftContactStats soft_stats_last;
	bool soft_contacts_enabled = false;
	bool soft_rigid_warned = false;
	bool soft_overflow_reported = false;
	bool soft_shear_enabled = true;

	// The last step's patch figures, kept only with physics/jolt_physics_3d/debug/soft_patch_figures on: fixed-size records
	// in room that grows only when a step has more patches than any before, so a step allocates nothing once warm
	struct SoftPatchRecord {
		JPH::BodyID body1;
		JPH::BodyID body2;
		int shape1 = -1;
		int shape2 = -1;
		float figures[JOLT_SOFT_PATCH_FIGURES] = {};
	};
	Mutex soft_records_mutex;
	LocalVector<SoftPatchRecord> soft_records;
	uint32_t soft_record_count = 0;

	// CRUMB (soft fix B, P5): the face laws kept across steps, with this space's own step counter (never another space's)
	JoltSoftFaceReuseCache soft_face_cache;

	void _pre_step(float p_step);
	void _post_step(float p_step);

	// CRUMB: everything a fresh physics world needs (settings, gravity, listeners, the collide and
	// combine callbacks), shared by the constructor and rebuild.
	void _init_physics_system();

public:
	explicit JoltSpace3D(JPH::JobSystem *p_job_system);
	~JoltSpace3D();

	void step(float p_step);

	// CRUMB: rebuild this space's physics world in place. p_objects are every object in the space,
	// in creation order: they leave the old world in reverse order and enter a fresh one in order,
	// so ids, broad phase and contact caches are those of a space that never held anything else.
	// The space's active flag is untouched. Refused (false, logged) while stepping, with any
	// constraint, or with a soft body.
	bool rebuild(const LocalVector<JoltObject3D *> &p_objects);

	void call_queries();

	RID get_rid() const { return rid; }
	void set_rid(const RID &p_rid) { rid = p_rid; }

	bool is_active() const { return active; }
	void set_active(bool p_active) { active = p_active; }

	bool is_stepping() const { return stepping; }

	double get_param(PhysicsServer3D::SpaceParameter p_param) const;
	void set_param(PhysicsServer3D::SpaceParameter p_param, double p_value);

	JPH::PhysicsSystem &get_physics_system() const { return *physics_system; }

	JPH::TempAllocator &get_temp_allocator() const { return *temp_allocator; }

	JPH::BodyInterface &get_body_iface();
	const JPH::BodyInterface &get_body_iface() const;
	const JPH::BodyLockInterface &get_lock_iface() const;

	const JPH::BroadPhaseQuery &get_broad_phase_query() const;
	const JPH::NarrowPhaseQuery &get_narrow_phase_query() const;

	JPH::ObjectLayer map_to_object_layer(JPH::BroadPhaseLayer p_broad_phase_layer, uint32_t p_collision_layer, uint32_t p_collision_mask);
	void map_from_object_layer(JPH::ObjectLayer p_object_layer, JPH::BroadPhaseLayer &r_broad_phase_layer, uint32_t &r_collision_layer, uint32_t &r_collision_mask) const;

	JPH::Body *try_get_jolt_body(const JPH::BodyID &p_body_id) const;
	JoltObject3D *try_get_object(const JPH::BodyID &p_body_id) const;
	JoltShapedObject3D *try_get_shaped(const JPH::BodyID &p_body_id) const;
	JoltBody3D *try_get_body(const JPH::BodyID &p_body_id) const;
	JoltArea3D *try_get_area(const JPH::BodyID &p_body_id) const;
	JoltSoftBody3D *try_get_soft_body(const JPH::BodyID &p_body_id) const;

	JoltPhysicsDirectSpaceState3D *get_direct_state();

	JoltArea3D *get_default_area() const { return default_area; }
	void set_default_area(JoltArea3D *p_area);

	float get_last_step() const { return last_step; }
	uint64_t get_step_serial() const { return step_serial; }

	// CRUMB: the soft contact. Soft contacts need single-threaded physics and at least 8 velocity steps; otherwise
	// they stay rigid, with one line. The counts and the patch records are the last completed step's.
	bool soft_surfaces_supported() const;
	bool are_soft_contacts_enabled() const { return soft_contacts_enabled; }
	bool is_soft_shear_enabled() const { return soft_shear_enabled; }
	// The kept face laws (physics thread only, inside the step)
	JoltSoftFaceReuseCache *get_soft_face_cache() { return &soft_face_cache; }
	void count_soft_contact(SoftCount p_count) { soft_counts[p_count].increment(); }
	// Whether the step keeps its patch figures (physics/jolt_physics_3d/debug/soft_patch_figures)
	static bool are_soft_patch_figures_kept();
	void record_soft_patch(const JPH::BodyID &p_body1, int p_shape1, const JPH::BodyID &p_body2, int p_shape2, const JPH::SoftContactPatch &p_patch, const JPH::Vec3 &p_normal, const JoltSoftBuildResult &p_result, int p_kind);
	PackedInt32Array get_soft_contact_stats() const;
	PackedFloat32Array get_soft_patch_debug(const JPH::BodyID &p_body_a, int p_shape_a, const JPH::BodyID &p_body_b, int p_shape_b) const;
	// The module's friction combine (the project setting's rule) of two figures
	static float combine_friction(float p_friction1, float p_friction2);

	JPH::Body *add_object(const JoltObject3D &p_object, const JPH::BodyCreationSettings &p_settings, bool p_sleeping = false);
	JPH::Body *add_object(const JoltObject3D &p_object, const JPH::SoftBodyCreationSettings &p_settings, bool p_sleeping = false);
	void remove_object(const JPH::BodyID &p_jolt_id);
	void flush_pending_objects();

	void set_is_object_sleeping(const JPH::BodyID &p_jolt_id, bool p_enable);

	void enqueue_call_queries(SelfList<JoltBody3D> *p_body);
	void enqueue_call_queries(SelfList<JoltArea3D> *p_area);
	void dequeue_call_queries(SelfList<JoltBody3D> *p_body);
	void dequeue_call_queries(SelfList<JoltArea3D> *p_area);

	void enqueue_shapes_changed(SelfList<JoltShapedObject3D> *p_object);
	void dequeue_shapes_changed(SelfList<JoltShapedObject3D> *p_object);

	void enqueue_needs_optimization(SelfList<JoltShapedObject3D> *p_object);
	void dequeue_needs_optimization(SelfList<JoltShapedObject3D> *p_object);

	void add_joint(JPH::Constraint *p_jolt_ref);
	void add_joint(JoltJoint3D *p_joint);
	void remove_joint(JPH::Constraint *p_jolt_ref);
	void remove_joint(JoltJoint3D *p_joint);

#ifdef DEBUG_ENABLED
	void dump_debug_snapshot(const String &p_dir);
	const PackedVector3Array &get_debug_contacts() const;
	int get_debug_contact_count() const;
	int get_max_debug_contacts() const;
	void set_max_debug_contacts(int p_count);
#endif
};
