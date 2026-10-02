/**************************************************************************/
/*  jolt_velocity_gear_joint_3d.h                                         */
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

#include "jolt_joint_3d.h"

#include "Jolt/Jolt.h"

// CRUMB: a relation among the spins of up to three bodies, each about an axis fixed in that body:
// c_a (w_a . a_a) + c_b (w_b . a_b) + c_c (w_c . a_c) = 0. A gear pair on a carrier that turns is
// (1, r, -(1 + r)); an open differential is (1, 1, -2); a geared drive's rotor, carrying the inertia
// it reflects to the output, is (1, -1, 1 - 1/N). Velocity only by default: the rotor and the
// differential have no phase anything physical depends on. Solved inside the step with the other
// constraints. Created through JoltPhysicsServer3D::velocity_gear_joint_create; body_c may be null for
// the two-body form.
//
// The phase hold (set_phase_hold, off by default) adds a position term for a mesh whose phase matters,
// such as teeth on a frame that turns: c_a t_a + c_b t_b = 0, where t_a and t_b are bodies a's and b's
// turns about their own axes since the relation first stepped, each measured relative to body c (to
// the world in the two-body form). The position step pulls it back to zero with Baumgarte, as a hinge
// corrects its drift, so whatever turns a body without a velocity the relation sees (another joint's
// or a contact's position correction, a teleport) no longer slips the mesh. It is the integral of the
// velocity relation when the axes are parallel and the coefficients sum to zero (a mesh on a carrier,
// a differential), which is where it belongs.
//
// The base joint registers bodies a and b; this one registers body c itself, so a body leaving or
// entering the space, any of the three, destroys or rebuilds the constraint. The constraint exists
// only while all three are in one space.
class JoltVelocityGearJoint3D final : public JoltJoint3D {
public:
	// The phase hold's switch and its tracking. The constraint keeps the tracking here, so a rebuilt
	// constraint (a body leaving and re-entering the space, a shape change) carries on from it instead
	// of starting again. A turn is read from the pose each time, as the twist about the body's axis of
	// its rotation since the reference, so it is a function of where the bodies are and never drifts;
	// once a step the constraint counts the whole turns it has passed, which reads true while a body
	// turns less than half a turn per step relative to body c. Steps the constraint takes no part in (a
	// body out of the space, the joint disabled) are bridged the same way, so a turn over such a gap
	// counts modulo a whole turn.
	struct PhaseState {
		bool hold = false;
		bool started = false;
		// Each body's rotation relative to body c (or the world) when the relation first stepped: x, y,
		// z, w.
		float reference[2][4] = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } };
		// Each body's turn from its reference at the last count, within half a turn either way, and the
		// whole turns counted past it.
		float last_turn[2] = { 0.0f, 0.0f };
		int whole_turns[2] = { 0, 0 };
	};

private:
	JoltBody3D *body_c = nullptr;

	Vector3 axis_a;
	Vector3 axis_b;
	Vector3 axis_c;
	Vector3 coefficients;

	PhaseState phase;

public:
	JoltVelocityGearJoint3D(const JoltJoint3D &p_old_joint, JoltBody3D *p_body_a, JoltBody3D *p_body_b, JoltBody3D *p_body_c, const Vector3 &p_axis_a, const Vector3 &p_axis_b, const Vector3 &p_axis_c, const Vector3 &p_coefficients);
	virtual ~JoltVelocityGearJoint3D() override;

	// No PhysicsServer3D::JointType fits; like the gear joint it is created and freed through the
	// server's RID owner directly, never by a Joint3D node.
	virtual PhysicsServer3D::JointType get_type() const override { return PhysicsServer3D::JOINT_TYPE_MAX; }

	bool is_phase_hold_enabled() const { return phase.hold; }
	void set_phase_hold(bool p_enable);

	virtual void rebuild() override;
};
