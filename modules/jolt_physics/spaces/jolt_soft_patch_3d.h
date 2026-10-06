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

#include "core/templates/local_vector.h"
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
// - Layer, Face against a partner that is not a Layer: the face law v2 (Stage 1c-b desk A, DESK_A.md 6): the layer's
//   film flow in conservative form on a tensor grid of cells (at most 0.35 depth across over the contactable region,
//   graded beyond it to a free layer four depths out, at most 1024 cells, its axes along the region's longest edge),
//   the contact set by complementarity, two fields (a frictionless face and the physical no-slip face, which holds each
//   point where it touched), the FE's edge multipliers cell by cell from the nearest free face and indenter edge, and
//   the Coulomb face as the FE's own friction share of the two fields (spaces/soft_layer_table/
//   jolt_soft_face_table.gen.h, its constants checked at compile time). Active-set passes from the tail's two sets,
//   banded Cholesky. A partner edge within 1e-3 depth of the layer's own edge is that free face (an exact coincidence
//   and a float's noise either side read alike); coplanar partner faces press as their union. The law carries the
//   bulk modulus itself (the compressible pad equation), so no separate compressibility term. Since the Stage 1c-b fix
//   round it is the face law v3 (the desk reference's section 6c): an indenter edge read by its margin (the free layer
//   between it and the layer's own edge) from the inset tables, its choke growing with the margin, so a part the pad's
//   size or a little smaller reads continuously from the free face to the punch; a soft choice between a free face and
//   an indenter edge of small margin where both are near; the grid turned to the mean direction of the region's
//   near-longest edges; embedded edges (each cell stands for the area really covered, the drain and the indenter edge
//   where the edges cross its axes), so round contacts read without the staircase's swing; at least 12 cells per axis;
//   and the held slide's memory in the tail (the gradient the contact grew at), so a face pressed flat and then tilted
//   makes no slide.
// - Layer, Line and Point: the layer table (spaces/soft_layer_table/jolt_soft_layer_table.gen.h, TABLE_FORMAT.v2):
//   a strip model of slices along a line, the axisymmetric figures at a point; outside the table, the split law (p =
//   (E/h) d + q, lam2 q - lap q = kI d, q = 0 at free edges) on a grid one depth beyond the contact with Richardson on
//   the half grid and T3 at the pair's static mu (counted).
// - Two Layer faces: the split law with the contact set by complementarity, each side's layer with a lubricated face,
//   compliances in series (the lead's decision 5).
// - Two Layer sides on a Line or a Point (Stage 1c-b fix round, C3): the lead's decision 5 through the table, each side's
//   layer with a lubricated face (the table's mu 0 column), compliances in series. Side i takes the share phi_i of the
//   pair's squash and sees the other side as a radius R / phi_i, phi_1 making the two forces equal (exact for identical
//   sides by the mirror symmetry; against fresh two-body FE within 1.3 %); a Line slice by slice. The shear rows are the
//   two layers in series. Where a side is outside the table, the side with the larger local share reads its own law at
//   that share (counted as outside the table).
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

// CRUMB (soft fix B, P5): the face law kept across steps. A face patch's law is rebuilt from scratch every step, and in a
// steady grip almost nothing it reads changes from one step to the next. Each Face patch of a Layer against a partner that
// is not a Layer keeps the law it last built, keyed by its manifold's bodies and sub shapes (ids, never addresses). While
// the patch's contact sets hold, the law at this step's plane is the kept one plus its tangent pressures times the plane's
// change (the fields are linear in the interference for a held set), so a step whose soft face moved rigidly, whose
// partner slid less than the slide limit against it, whose held slide's memory is the kept one's and whose cells keep
// their classes reuses it: its pressures moved along their own tangents, complementarity checked on every contactable
// cell, the friction share read again at this step's tau, the figures made by the law's own figure block in this step's
// frame, and the held slide's memory updated as the law updates it. Anything else rebuilds (see f2_face_reuse), as does
// the 33rd reuse in a row, a first touch still to be decided, and a tail other than the one the kept law wrote. An entry
// untouched for a step is dropped when the step ends; a rebuild of the space drops them all. The physics thread alone
// reads and writes it (soft contacts need single-threaded Jolt).
// Switches, read when a world is made: CRUMB_SOFT_FACE_REUSE_UM, the slide limit in micrometres (default 8; 0 turns the
// reuse off and every step builds the law as before); CRUMB_SOFT_FACE_REUSE_LOG=1 prints the counts; CRUMB_SOFT_FACE_REUSE_CHECK=1
// also builds the law beside every reuse, into scratch the patch never reads, and prints how far the reuse was from it.
struct JoltSoftFaceReuseEntry; // the builder's own (jolt_soft_patch_3d.cpp)

class JoltSoftFaceReuseCache {
public:
	struct Key {
		uint32_t body1 = 0;
		uint32_t sub1 = 0;
		uint32_t body2 = 0;
		uint32_t sub2 = 0;

		bool operator==(const Key &p_other) const { return body1 == p_other.body1 && sub1 == p_other.sub1 && body2 == p_other.body2 && sub2 == p_other.sub2; }
	};

	// What each face law call did: reused, or built and why
	enum Outcome {
		REUSED,
		BUILT_NO_ENTRY, // new, or not written in the last step
		BUILT_TWICE, // a second call for one manifold in one step (kept no longer)
		BUILT_ONSET, // the first touch is still to be decided
		BUILT_UNSETTLED, // the kept law's sets did not settle, or it had no held slide's memory
		BUILT_TAIL, // the tail is not the one the kept law wrote
		BUILT_COUNTS, // the material, the friction or a face's vertex count changed
		BUILT_CAP, // reused REUSE_CAP times in a row
		BUILT_SHAPE, // the soft face is not the kept one moved rigidly
		BUILT_SLIDE, // the partner slid more than the limit against it
		BUILT_HELD, // the held slide's memory is not the kept one's
		BUILT_CELLS, // a contactable cell changed class
		BUILT_PHI, // the frictionless field's solve flips
		BUILT_COMPL, // complementarity fails on a contactable cell
		BUILT_FIGURES, // the reused figures press nothing
		OUTCOME_MAX,
	};
	static constexpr int REUSE_CAP = 32;

	uint64_t outcomes[OUTCOME_MAX] = {};
	// CRUMB_SOFT_FACE_REUSE_CHECK: the reuses against the law built beside them, the force at least 0.5 N (light contacts
	// counted apart). agree: the law's contact sets are the reuse's; differ: they are not.
	struct CheckStats {
		uint64_t checked = 0;
		uint64_t light = 0;
		double light_df_max = 0.0; // |dF|, solver units
		uint64_t agree = 0;
		uint64_t agree_over = 0; // over the desk's bound: |dF| / F above 4e-3 or the centre more than 8 um off
		double agree_df_max = 0.0, agree_c_max = 0.0, agree_arm_max = 0.0, agree_kt_max = 0.0; // relative, mm, mm, relative
		uint64_t differ = 0;
		double differ_df_max = 0.0, differ_c_max = 0.0, differ_arm_max = 0.0;
		uint64_t differ_within_step = 0; // |dF| no more than the law's own change from the last step's figures
		uint64_t differ_within_build = 0; // |dF| no more than the law's own change since the kept law was built
	} check_stats;

	~JoltSoftFaceReuseCache();

	// Reads the switches (a world is made)
	void configure();
	bool is_enabled() const { return slide_limit > 0.0; }
	bool is_checking() const { return check; }
	double get_slide_limit() const { return slide_limit; } // mm
	uint32_t get_step() const { return step; }

	// The space's own step counter (never another space's), and the drop of entries untouched in the step
	void begin_step();
	void end_step();
	// Drops every entry (a rebuild of the space: its bodies take the same ids again)
	void clear();
	// The entry of one manifold, made when it has none; nullptr when the reuse is off. Marks it touched in this step.
	JoltSoftFaceReuseEntry *touch(const Key &p_key);
	// The builder counts each call (and prints the first reuse)
	void count(Outcome p_outcome);

private:
	struct Slot {
		Key key;
		JoltSoftFaceReuseEntry *entry = nullptr;
		uint32_t touched = 0;
	};
	LocalVector<Slot> slots;
	LocalVector<JoltSoftFaceReuseEntry *> spare; // dropped entries' room, kept for the next new pair
	double slide_limit = 0.0;
	bool log = false;
	bool check = false;
	bool reused_once = false;
	uint32_t step = 0;
	uint32_t active_steps = 0; // steps with a face law call
	bool active_now = false;

	void _print(const char *p_when) const;
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
	JoltSoftFaceReuseCache *face_cache = nullptr; // the space's kept face laws (nullptr: built every step)
	JoltSoftFaceReuseCache::Key face_key; // this manifold's key in it
};

// Builds one manifold's patch. Returns true with r_patch enabled when the contact is soft this step.
bool jolt_build_soft_patch(const JoltSoftPatchInput &p_in, JPH::SoftContactPatch &r_patch, JoltSoftBuildResult &r_result);

// The patch figures as space_debug_soft_patch returns them (documented at JoltPhysicsServer3D::space_debug_soft_patch),
// written into r_figures (JOLT_SOFT_PATCH_FIGURES floats); no allocation
constexpr int JOLT_SOFT_PATCH_FIGURES = 40;
void jolt_soft_patch_debug_figures(const JPH::SoftContactPatch &p_patch, const JPH::Vec3 &p_normal, const JoltSoftBuildResult &p_result, int p_kind, float *r_figures);
