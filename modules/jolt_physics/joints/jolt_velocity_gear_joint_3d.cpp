/**************************************************************************/
/*  jolt_velocity_gear_joint_3d.cpp                                       */
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

#include "jolt_velocity_gear_joint_3d.h"

#include "../misc/jolt_type_conversions.h"
#include "../objects/jolt_body_3d.h"
#include "../spaces/jolt_space_3d.h"

#include "Jolt/Physics/Body/Body.h"
#include "Jolt/Physics/Body/BodyManager.h"
#include "Jolt/Physics/Constraints/Constraint.h"
#include "Jolt/Physics/IslandBuilder.h"
#include "Jolt/Physics/LargeIslandSplitter.h"
#include "Jolt/Physics/StateRecorder.h"

namespace {

// The settings a debug dump asks a constraint for. Nothing creates the constraint from it: the
// joint builds the constraint directly.
class CrumbVelocityGearConstraintSettings final : public JPH::ConstraintSettings {
public:
	JPH_OVERRIDE_NEW_DELETE

	CrumbVelocityGearConstraintSettings() = default;
};

// One scalar row over up to three bodies: the sum of c_i (w_i . a_i), with a_i each body's axis turned
// into the world at the start of the step. A static body contributes nothing, a kinematic body its
// own spin; only dynamic bodies take impulses. The position solve does nothing unless the joint's phase
// hold is on (see JoltVelocityGearJoint3D); the turns it holds are tracked every step either way.
class CrumbVelocityGearConstraint final : public JPH::Constraint {
	JPH::Body *bodies[3] = { nullptr, nullptr, nullptr };
	JPH::Vec3 local_axes[3];
	float coefficients[3] = { 0.0f, 0.0f, 0.0f };
	int body_count = 0;

	// The joint's phase hold: its switch and the tracking, which outlive this constraint.
	JoltVelocityGearJoint3D::PhaseState *phase = nullptr;

	// Set up once per step: each body's row, c_i R_i a_i, and I_i^-1 of it for a dynamic body.
	JPH::Vec3 rows[3];
	JPH::Vec3 inv_inertia_rows[3];
	float effective_mass = 0.0f;
	float total_lambda = 0.0f;

	void _apply_impulse(float p_lambda) {
		for (int i = 0; i < body_count; ++i) {
			JPH::Body &body = *bodies[i];
			if (body.IsDynamic()) {
				body.GetMotionPropertiesUnchecked()->AddAngularVelocityStep(p_lambda * inv_inertia_rows[i]);
			}
		}
	}

	// Body a's or b's rotation relative to body c, or to the world in the two-body form.
	JPH::Quat _relative_rotation(int p_index) const {
		const JPH::Quat rotation = bodies[p_index]->GetRotation();
		return body_count == 3 ? bodies[2]->GetRotation().Conjugated() * rotation : rotation;
	}

	// Body a's or b's turn about its own axis from its reference, from the pose: the twist of its
	// rotation since the reference, read from the half of the double cover with w >= 0, so it lies
	// within half a turn either way.
	float _turn_from_reference(int p_index) const {
		const float *reference = phase->reference[p_index];
		const JPH::Quat change = JPH::Quat(reference[0], reference[1], reference[2], reference[3]).Conjugated() * _relative_rotation(p_index);
		float s = change.GetXYZ().Dot(local_axes[p_index]);
		float w = change.GetW();
		if (w < 0.0f) {
			s = -s;
			w = -w;
		}
		return 2.0f * Math::atan2(s, w);
	}

	// The turn since the last count, taken within half a turn either way.
	float _turn_since_count(int p_index, float p_turn) const {
		float change = p_turn - phase->last_turn[p_index];
		if (change > (float)Math::PI) {
			change -= (float)Math::TAU;
		} else if (change < (float)-Math::PI) {
			change += (float)Math::TAU;
		}
		return change;
	}

	// Body a's or b's whole turn from its reference now: the whole turns counted, the turn at the last
	// count, and the turn since.
	double _turn(int p_index) const {
		const float since = _turn_since_count(p_index, _turn_from_reference(p_index));
		return Math::TAU * phase->whole_turns[p_index] + (double)phase->last_turn[p_index] + (double)since;
	}

	// Once per step, before the solve: counts the whole turns each body has passed since the last count.
	// The first step takes the references instead, so the turns count from the poses the relation first
	// steps from.
	void _track_phase() {
		for (int i = 0; i < 2; ++i) {
			if (!phase->started) {
				const JPH::Quat relative = _relative_rotation(i);
				float *reference = phase->reference[i];
				reference[0] = relative.GetX();
				reference[1] = relative.GetY();
				reference[2] = relative.GetZ();
				reference[3] = relative.GetW();
				phase->last_turn[i] = 0.0f;
				phase->whole_turns[i] = 0;
				continue;
			}
			const float turn = _turn_from_reference(i);
			const float raw_change = turn - phase->last_turn[i];
			if (raw_change > (float)Math::PI) {
				phase->whole_turns[i] -= 1;
			} else if (raw_change < (float)-Math::PI) {
				phase->whole_turns[i] += 1;
			}
			phase->last_turn[i] = turn;
		}
		phase->started = true;
	}

public:
	CrumbVelocityGearConstraint(const JPH::ConstraintSettings &p_settings, JPH::Body *const p_bodies[3], const JPH::Vec3 p_local_axes[3], const float p_coefficients[3], int p_body_count, JoltVelocityGearJoint3D::PhaseState *p_phase) :
			JPH::Constraint(p_settings) {
		phase = p_phase;
		body_count = p_body_count;
		for (int i = 0; i < 3; ++i) {
			bodies[i] = i < body_count ? p_bodies[i] : nullptr;
			local_axes[i] = p_local_axes[i];
			coefficients[i] = i < body_count ? p_coefficients[i] : 0.0f;
			rows[i] = JPH::Vec3::sZero();
			inv_inertia_rows[i] = JPH::Vec3::sZero();
		}
	}

	virtual JPH::EConstraintSubType GetSubType() const override { return JPH::EConstraintSubType::User1; }

	// The axes are directions fixed in each body, so a shape change that moves a centre of mass
	// changes nothing here.
	virtual void NotifyShapeChanged(const JPH::BodyID &, JPH::Vec3Arg) override {}

	virtual void ResetWarmStart() override { total_lambda = 0.0f; }

	// As a two-body constraint: enabled, at least one body moving, at least one body dynamic.
	virtual bool IsActive() const override {
		if (!JPH::Constraint::IsActive()) {
			return false;
		}
		bool any_active = false;
		bool any_dynamic = false;
		for (int i = 0; i < body_count; ++i) {
			any_active = any_active || bodies[i]->IsActive();
			any_dynamic = any_dynamic || bodies[i]->IsDynamic();
		}
		return any_active && any_dynamic;
	}

	virtual void SetupVelocityConstraint(float) override {
		float k = 0.0f;
		for (int i = 0; i < body_count; ++i) {
			const JPH::Body &body = *bodies[i];
			rows[i] = coefficients[i] * (body.GetRotation() * local_axes[i]);
			if (body.IsDynamic()) {
				inv_inertia_rows[i] = body.GetMotionPropertiesUnchecked()->MultiplyWorldSpaceInverseInertiaByVector(body.GetRotation(), rows[i]);
				k += rows[i].Dot(inv_inertia_rows[i]);
			} else {
				inv_inertia_rows[i] = JPH::Vec3::sZero();
			}
		}
		effective_mass = k > 0.0f ? 1.0f / k : 0.0f;

		// Reads the poses and writes only the tracking, so with the hold off a scene steps bit for bit
		// as without it.
		_track_phase();
	}

	virtual void WarmStartVelocityConstraint(float p_warm_start_impulse_ratio) override {
		total_lambda *= p_warm_start_impulse_ratio;
		if (effective_mass == 0.0f) {
			total_lambda = 0.0f;
			return;
		}
		if (total_lambda != 0.0f) {
			_apply_impulse(total_lambda);
		}
	}

	virtual bool SolveVelocityConstraint(float) override {
		if (effective_mass == 0.0f) {
			return false;
		}
		float jv = 0.0f;
		for (int i = 0; i < body_count; ++i) {
			jv += rows[i].Dot(bodies[i]->GetAngularVelocity());
		}
		const float lambda = -effective_mass * jv;
		if (lambda == 0.0f) {
			return false;
		}
		total_lambda += lambda;
		_apply_impulse(lambda);
		return true;
	}

	// The phase hold: C = c_a t_a + c_b t_b, the turns read from the poses as the position step has left
	// them. Its gradient is c_a n_a on body a and c_b n_b on body b, n the axes in the world, and minus
	// their sum on body c, whose turn moves both measures; with parallel axes and coefficients summing
	// to zero that is the velocity row itself. Corrected as Jolt's own position parts do:
	// lambda = -beta C / K, each dynamic body turned by lambda I^-1 J with no velocity left behind, so the
	// hold adds no momentum.
	virtual bool SolvePositionConstraint(float, float p_baumgarte) override {
		if (!phase->hold || !phase->started) {
			return false;
		}

		// In double: a mesh that has run many turns holds the small difference of two large turns.
		const float error = (float)(coefficients[0] * _turn(0) + coefficients[1] * _turn(1));
		if (error == 0.0f) {
			return false;
		}

		JPH::Vec3 jacobian[3];
		for (int i = 0; i < 2; ++i) {
			jacobian[i] = coefficients[i] * (bodies[i]->GetRotation() * local_axes[i]);
		}
		jacobian[2] = -(jacobian[0] + jacobian[1]);

		float k = 0.0f;
		JPH::Vec3 inv_inertia_jacobian[3];
		for (int i = 0; i < body_count; ++i) {
			const JPH::Body &body = *bodies[i];
			if (body.IsDynamic()) {
				inv_inertia_jacobian[i] = body.GetMotionPropertiesUnchecked()->MultiplyWorldSpaceInverseInertiaByVector(body.GetRotation(), jacobian[i]);
				k += jacobian[i].Dot(inv_inertia_jacobian[i]);
			} else {
				inv_inertia_jacobian[i] = JPH::Vec3::sZero();
			}
		}
		if (k <= 0.0f) {
			return false;
		}

		const float lambda = -p_baumgarte * error / k;
		for (int i = 0; i < body_count; ++i) {
			if (bodies[i]->IsDynamic()) {
				bodies[i]->AddRotationStep(lambda * inv_inertia_jacobian[i]);
			}
		}
		return true;
	}

	// Wake every dynamic body the relation ties, then link them all into one island. The constraint
	// is linked to the lowest active index, never to two inactive ones.
	virtual void BuildIslands(JPH::uint32 p_constraint_index, JPH::IslandBuilder &r_builder, JPH::BodyManager &p_body_manager) override {
		JPH::BodyID to_activate[3];
		int activate_count = 0;
		for (int i = 0; i < body_count; ++i) {
			if (bodies[i]->IsDynamic() && !bodies[i]->IsActive()) {
				to_activate[activate_count++] = bodies[i]->GetID();
			}
		}
		if (activate_count > 0) {
			p_body_manager.ActivateBodies(to_activate, activate_count);
		}

		JPH::uint32 lowest = JPH::Body::cInactiveIndex;
		for (int i = 0; i < body_count; ++i) {
			lowest = MIN(lowest, bodies[i]->GetIndexInActiveBodiesInternal());
		}
		for (int i = 0; i < body_count; ++i) {
			const JPH::uint32 index = bodies[i]->GetIndexInActiveBodiesInternal();
			if (index != JPH::Body::cInactiveIndex && index != lowest) {
				r_builder.LinkBodies(lowest, index);
			}
		}
		r_builder.LinkConstraint(p_constraint_index, lowest, lowest);
	}

	virtual JPH::uint BuildIslandSplits(JPH::LargeIslandSplitter &r_splitter) const override {
		for (int i = 0; i < body_count; ++i) {
			r_splitter.AssignToNonParallelSplit(bodies[i]);
		}
		return JPH::LargeIslandSplitter::cNonParallelSplitIdx;
	}

#ifdef JPH_DEBUG_RENDERER
	virtual void DrawConstraint(JPH::DebugRenderer *) const override {}
#endif

	// The phase tracking is state as much as the warm start is; the hold's switch is the joint's setting.
	virtual void SaveState(JPH::StateRecorder &r_stream) const override {
		JPH::Constraint::SaveState(r_stream);
		r_stream.Write(total_lambda);
		r_stream.Write(phase->started);
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 4; ++j) {
				r_stream.Write(phase->reference[i][j]);
			}
			r_stream.Write(phase->last_turn[i]);
			r_stream.Write(phase->whole_turns[i]);
		}
	}

	virtual void RestoreState(JPH::StateRecorder &r_stream) override {
		JPH::Constraint::RestoreState(r_stream);
		r_stream.Read(total_lambda);
		r_stream.Read(phase->started);
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 4; ++j) {
				r_stream.Read(phase->reference[i][j]);
			}
			r_stream.Read(phase->last_turn[i]);
			r_stream.Read(phase->whole_turns[i]);
		}
	}

	virtual JPH::Ref<JPH::ConstraintSettings> GetConstraintSettings() const override {
		CrumbVelocityGearConstraintSettings *settings = new CrumbVelocityGearConstraintSettings();
		ToConstraintSettings(*settings);
		return settings;
	}
};

} // namespace

JoltVelocityGearJoint3D::JoltVelocityGearJoint3D(const JoltJoint3D &p_old_joint, JoltBody3D *p_body_a, JoltBody3D *p_body_b, JoltBody3D *p_body_c, const Vector3 &p_axis_a, const Vector3 &p_axis_b, const Vector3 &p_axis_c, const Vector3 &p_coefficients) :
		JoltJoint3D(p_old_joint, p_body_a, p_body_b, Transform3D(), Transform3D()),
		body_c(p_body_c),
		axis_a(p_axis_a.normalized()),
		axis_b(p_axis_b.normalized()),
		axis_c(p_body_c != nullptr ? p_axis_c.normalized() : Vector3()),
		coefficients(p_coefficients) {
	if (body_c != nullptr) {
		body_c->add_joint(this);
	}

	rebuild();
}

JoltVelocityGearJoint3D::~JoltVelocityGearJoint3D() {
	// Out of the world while body c is still registered, then let body c go; the base lets a and b go.
	destroy();

	if (body_c != nullptr) {
		body_c->remove_joint(this);
	}
}

void JoltVelocityGearJoint3D::rebuild() {
	destroy();

	JoltSpace3D *space = get_space();
	if (space == nullptr) {
		return;
	}

	// All three bodies must be in this one space; while any is out the relation is down, and the
	// body's return rebuilds it.
	if (body_c != nullptr && body_c->get_space() != space) {
		return;
	}

	JPH::Body *jolt_bodies[3] = {
		body_a != nullptr ? body_a->get_jolt_body() : nullptr,
		body_b != nullptr ? body_b->get_jolt_body() : nullptr,
		body_c != nullptr ? body_c->get_jolt_body() : nullptr,
	};
	if (jolt_bodies[0] == nullptr || jolt_bodies[1] == nullptr || (body_c != nullptr && jolt_bodies[2] == nullptr)) {
		return;
	}

	const JPH::Vec3 local_axes[3] = { to_jolt(axis_a), to_jolt(axis_b), to_jolt(axis_c) };
	const float coefficient_values[3] = { (float)coefficients.x, (float)coefficients.y, (float)coefficients.z };

	CrumbVelocityGearConstraintSettings settings;
	jolt_ref = new CrumbVelocityGearConstraint(settings, jolt_bodies, local_axes, coefficient_values, body_c != nullptr ? 3 : 2, &phase);

	space->add_joint(this);

	_update_enabled();
	_update_iterations();
}

void JoltVelocityGearJoint3D::set_phase_hold(bool p_enable) {
	if (phase.hold == p_enable) {
		return;
	}

	// Held relative to body c, the phase is the relation's integral only when the coefficients sum to
	// zero (with the axes parallel); otherwise the two pull against each other whenever body c turns.
	if (p_enable && body_c != nullptr) {
		const real_t sum = coefficients.x + coefficients.y + coefficients.z;
		const real_t scale = MAX(Math::abs(coefficients.x), MAX(Math::abs(coefficients.y), Math::abs(coefficients.z)));
		if (Math::abs(sum) > (real_t)1e-4 * scale) {
			WARN_PRINT(vformat("velocity_gear_joint_set_phase_hold: the coefficients %s do not sum to zero, so the phase it holds (the turns of bodies a and b relative to body c) is not the relation's integral, and the two pull against each other whenever body c turns.", coefficients));
		}
	}

	phase.hold = p_enable;

	_wake_up_bodies();
	if (body_c != nullptr) {
		body_c->wake_up();
	}
}
