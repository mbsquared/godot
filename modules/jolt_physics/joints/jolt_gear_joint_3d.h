/**************************************************************************/
/*  jolt_gear_joint_3d.h                                                  */
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

#include "../jolt_physics_server_3d.h"
#include "jolt_joint_3d.h"

#include "Jolt/Jolt.h"

// A GEAR coupling between two rotating bodies: a hard velocity constraint
// omega_a . axis_a = -ratio * omega_b . axis_b, solved by Jolt every step, so a braked
// output rigidly stalls the input (no slip, no penalty wind-up, no release kick — the
// failure modes of the hand-rolled PD coupling this replaces). Axes are each body's spin
// axis in its OWN local frame; ratio is teeth_target / teeth_owner (negate to flip mesh
// direction: belt / internal ring gear). Created via JoltPhysicsServer3D::gear_joint_create.
class JoltGearJoint3D final : public JoltJoint3D {
	Vector3 hinge_axis_a;
	Vector3 hinge_axis_b;
	float ratio = 1.0f;

public:
	JoltGearJoint3D(const JoltJoint3D &p_old_joint, JoltBody3D *p_body_a, JoltBody3D *p_body_b, const Vector3 &p_axis_a, const Vector3 &p_axis_b, float p_ratio);

	// No dedicated PhysicsServer3D::JointType exists for gears; this joint is created + freed through the
	// server's RID owner directly (never a Joint3D node), so the type is informational only.
	virtual PhysicsServer3D::JointType get_type() const override { return PhysicsServer3D::JOINT_TYPE_MAX; }

	virtual void rebuild() override;
};
