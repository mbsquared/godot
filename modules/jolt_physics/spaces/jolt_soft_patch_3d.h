/**************************************************************************/
/*  jolt_soft_patch_3d.h                                                  */
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

#include "core/variant/variant.h"

#include "Jolt/Jolt.h"

#include "Jolt/Physics/Body/Body.h"
#include "Jolt/Physics/Collision/ContactListener.h"

// CRUMB: the compliant (soft) contact's patch builder (SOFT-CONTACT brief 3, 5.1 to 5.3; COMPLIANT design 4.1 and
// 4.2 where the brief is silent). A pure function of the two bodies, their shapes' soft surfaces and the manifold:
// it finds both sides' supporting faces along the manifold normal (never the pruned contact points), classes the
// contact (Face, Line or Point), builds the overlap field and the pressure field of the surface's law, and writes the
// patch figures the solver's rows need into a JPH::SoftContactPatch. Fixed sizes, a fixed order of operations, no
// allocation: the same inputs give the same bits.
//
// The laws:
// - Layer, Face: the split law (p = (E/h) d + q, lam2 q - lap q = kI d, q = 0 at free edges) with the contact set
//   found by complementarity on a grid extended one layer depth beyond the overlap, banded Cholesky, active-set
//   passes from the tail's set, Richardson on the half grid, scaled by the layer table's T3 at the pair's static mu.
//   The law carries the bulk modulus itself (the compressible pad equation), so no separate compressibility term.
// - Layer, Line and Point: the layer table (spaces/soft_layer_table/jolt_soft_layer_table.gen.h, TABLE_FORMAT.v2):
//   a strip model of slices along a line, the axisymmetric figures at a point; outside the table, the split law on
//   the grid (counted).
// - Two Layer faces: each side's layer with a lubricated face, compliances in series (the lead's decision 5).
// - Two Layer sides on a Line or a Point: the two-sided design's law (design 4.6: the softer side's split law with the
//   layers' local terms in series) on the grid, counted as outside the table. A disclosed restriction: the lubricated
//   series of decision 5 is not read for curved patches, since the table holds one layer's figures and a series of two
//   through it has no desk reference yet; PS7's band layer2 cases gate the law as built.
// - Solid: Hertz at a point, the line law of design 2.3 (UNSOURCED constants) along a line; a Solid face pair stays
//   rigid (counted).
// - Lumped: F = K_l d with the class's bed-of-springs profile.
// Each patch also carries its first-touch law as samples of its own pressed law (the solver reads them piecewise), the
// point where a first touch solved as an impact acts (a Line: the centroid of the overlap one speculative band deeper
// than its deepest point), and its core points (a face's overlap vertices, a Line's loaded ends, a Point's deepest point).
// Units are the solver's: millimetres, kilograms, seconds (a force of 1 N is 1000).

class JoltBody3D;

// One shape's soft surface as the host sends it (PhysicsServer3D::body_set_shape_soft_surface), in solver units
struct JoltSoftSurface3D {
	enum Kind {
		KIND_NONE = 0,
		KIND_LAYER = 1,
		KIND_SOLID = 2,
		KIND_LUMPED = 3,
	};

	int kind = KIND_NONE;
	double modulus = 0.0; // E
	double bulk_modulus = 0.0; // K
	double thickness = 0.0; // the layer's depth from the touching face to the rigid backing
	double lumped_stiffness = 0.0; // per contact, against one rigid flat
	double travel = 0.0; // the deflection at which a lumped body bottoms out (0: none)
	double zeta[3] = { 0.0, 0.0, 0.0 }; // damping ratios for force laws of exponent 1, 1.5 and 2
	double loss = 0.0; // the hysteresis loss fraction, 1 - resilience
	double restitution = 0.0; // sqrt(resilience)
	double sliding_friction = -1.0; // the kinetic coefficient; negative: the static one
	double rolling_arm = 0.0; // the measured rolling resistance arm, added to the layer's (appended, F13b)

	// Reads the parameter array; false with r_error when it cannot be a soft surface (an empty array is KIND_NONE)
	static bool from_params(const PackedFloat32Array &p_params, JoltSoftSurface3D &r_surface, String &r_error);

	// The damping ratio for a law of this exponent, linear between 1, 1.5 and 2 and clamped to them
	double zeta_for(double p_exponent) const;
};

// What the builder did with one manifold, for the status counts
struct JoltSoftBuildResult {
	enum Outcome {
		BUILT = 0, // the patch is enabled
		UNCLASSED, // the pair's geometry is none of Face, Line or Point: rigid
		SOLID_FACE, // a Solid face pair: rigid by design
		NO_OVERLAP, // the two footprints do not overlap in the patch plane: rigid
	};

	Outcome outcome = UNCLASSED;
	bool outside_table = false; // a curved patch the table does not cover: the split law on the grid
	bool not_settled = false; // a face patch's contact set did not settle within the pass cap
	int passes = 0; // active-set passes of the face grid
	double face_ratio = 1.0; // the face condition's ratio applied (T3), 1 when none
	double raw_force = 0.0; // the field's force before the face ratio
	double m1[2] = { 0.0, 0.0 }; // the field's first moments along the patch axes
};

struct JoltSoftPatchInput {
	const JPH::Body *body1 = nullptr;
	const JPH::Body *body2 = nullptr;
	const JPH::ContactManifold *manifold = nullptr;
	const JoltSoftSurface3D *surface1 = nullptr; // nullptr: that side is rigid
	const JoltSoftSurface3D *surface2 = nullptr;
	float static_friction = 0.0f; // the pair's combined static coefficient
	float sliding_friction = 0.0f; // the pair's combined kinetic coefficient (never above the static one)
	const JPH::SoftContactTail *tail = nullptr; // last step's tail of this manifold
};

// Builds one manifold's patch. Returns true with r_patch enabled when the contact is soft this step.
bool jolt_build_soft_patch(const JoltSoftPatchInput &p_in, JPH::SoftContactPatch &r_patch, JoltSoftBuildResult &r_result);

// The patch figures as space_debug_soft_patch returns them (documented at JoltPhysicsServer3D::space_debug_soft_patch)
PackedFloat32Array jolt_soft_patch_debug_array(const JPH::SoftContactPatch &p_patch, const JPH::Vec3 &p_normal, const JoltSoftBuildResult &p_result, int p_kind);
