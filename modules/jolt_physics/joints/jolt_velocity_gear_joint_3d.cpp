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
// own spin; only dynamic bodies take impulses. Velocity only: the position solve does nothing.
class CrumbVelocityGearConstraint final : public JPH::Constraint {
	JPH::Body *bodies[3] = { nullptr, nullptr, nullptr };
	JPH::Vec3 local_axes[3];
	float coefficients[3] = { 0.0f, 0.0f, 0.0f };
	int body_count = 0;

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

public:
	CrumbVelocityGearConstraint(const JPH::ConstraintSettings &p_settings, JPH::Body *const p_bodies[3], const JPH::Vec3 p_local_axes[3], const float p_coefficients[3], int p_body_count) :
			JPH::Constraint(p_settings) {
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

	virtual bool SolvePositionConstraint(float, float) override { return false; }

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

	virtual void SaveState(JPH::StateRecorder &r_stream) const override {
		JPH::Constraint::SaveState(r_stream);
		r_stream.Write(total_lambda);
	}

	virtual void RestoreState(JPH::StateRecorder &r_stream) override {
		JPH::Constraint::RestoreState(r_stream);
		r_stream.Read(total_lambda);
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
	jolt_ref = new CrumbVelocityGearConstraint(settings, jolt_bodies, local_axes, coefficient_values, body_c != nullptr ? 3 : 2);

	space->add_joint(this);

	_update_enabled();
	_update_iterations();
}
