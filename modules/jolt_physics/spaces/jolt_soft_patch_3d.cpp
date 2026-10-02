/**************************************************************************/
/*  jolt_soft_patch_3d.cpp                                                */
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

#include "jolt_soft_patch_3d.h"

#include "soft_layer_table/jolt_soft_layer_table.gen.h"

#include "../jolt_project_settings.h"

#include "Jolt/Geometry/AABox.h"
#include "Jolt/Physics/Collision/Shape/CapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/CompoundShape.h"
#include "Jolt/Physics/Collision/Shape/CylinderShape.h"
#include "Jolt/Physics/Collision/Shape/DecoratedShape.h"
#include "Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h"
#include "Jolt/Physics/Collision/Shape/ScaledShape.h"
#include "Jolt/Physics/Collision/Shape/SphereShape.h"

#include <cmath>

namespace {

namespace table = jolt_soft_layer_table;
static_assert(table::VERSION == 2, "The module reads version 2 of the soft layer table (TABLE_FORMAT.v2).");

constexpr int N_SHORT = 8; // grid cells across the short axis (design 4.2)
constexpr int N_LONG_MAX = 24; // and at most this many along (192 cells)
constexpr int MAX_CELLS = N_LONG_MAX * N_SHORT;
constexpr int ACTIVE_SET_CAP = 8; // active-set passes (brief 3.1)
constexpr int N_SLICES = 24; // slices of a Line patch's strip model
constexpr int MAX_POLY = 72; // a supporting face has at most 32 vertices; clipping adds at most one per edge
constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------------------------------------------
// The split law's coefficients (design 2.2): p = (E/h) w + q, lam2 q - lap q = kI w, q = 0 on free edges. kI is
// 12 G / h^3 for a contact face that does not slip, 3 G / h^3 for a lubricated face.

struct Coeffs {
	double Eh = 0.0;
	double kI = 0.0;
	double lam2 = 0.0;
	double G = 0.0;
	double nu = 0.0;
	double E = 0.0;
	double K = 0.0;
	double h = 0.0;
};

void elastic(double p_E, double p_K, double &r_G, double &r_nu) {
	r_G = 3.0 * p_K * p_E / (9.0 * p_K - p_E);
	r_nu = (3.0 * p_K - 2.0 * r_G) / (2.0 * (3.0 * p_K + r_G));
}

Coeffs split_coeffs(double p_E, double p_K, double p_h, bool p_lubricated) {
	Coeffs co;
	elastic(p_E, p_K, co.G, co.nu);
	const double kW = p_E * (1.0 - co.nu) / ((1.0 + co.nu) * (1.0 - 2.0 * co.nu) * p_h);
	co.kI = (p_lubricated ? 3.0 : 12.0) * co.G / (p_h * p_h * p_h);
	const double kq = kW - p_E / p_h;
	co.lam2 = kq > 1e-12 ? co.kI / kq : 0.0;
	co.Eh = p_E / p_h;
	co.E = p_E;
	co.K = p_K;
	co.h = p_h;
	return co;
}

// Design 4.6's two soft sides (kept for curved patches): the local terms in series, the bulging term from the side
// with the lower E / h
Coeffs two_sided_design(const JoltSoftSurface3D &p_s1, const JoltSoftSurface3D &p_s2) {
	const bool first_softer = p_s1.modulus / p_s1.thickness <= p_s2.modulus / p_s2.thickness;
	const JoltSoftSurface3D &soft = first_softer ? p_s1 : p_s2;
	Coeffs co = split_coeffs(soft.modulus, soft.bulk_modulus, soft.thickness, false);
	co.Eh = 1.0 / (p_s1.thickness / p_s1.modulus + p_s2.thickness / p_s2.modulus);
	return co;
}

// ---------------------------------------------------------------------------------------------------------------
// Convex polygons in the patch plane

struct Poly {
	int n = 0;
	double x[MAX_POLY];
	double y[MAX_POLY];

	void push(double p_x, double p_y) {
		if (n < MAX_POLY) {
			x[n] = p_x;
			y[n] = p_y;
			++n;
		}
	}
};

double signed_area(const Poly &p_poly) {
	double a = 0.0;
	for (int k = 0; k < p_poly.n; ++k) {
		const int k1 = (k + 1) % p_poly.n;
		a += p_poly.x[k] * p_poly.y[k1] - p_poly.x[k1] * p_poly.y[k];
	}
	return 0.5 * a;
}

void make_ccw(Poly &r_poly) {
	if (signed_area(r_poly) >= 0.0) {
		return;
	}
	for (int k = 0; k < r_poly.n / 2; ++k) {
		SWAP(r_poly.x[k], r_poly.x[r_poly.n - 1 - k]);
		SWAP(r_poly.y[k], r_poly.y[r_poly.n - 1 - k]);
	}
}

// Keep a x + b y + c > 0 (one edge of Sutherland and Hodgman)
void clip_halfplane(const Poly &p_in, double p_a, double p_b, double p_c, Poly &r_out) {
	r_out.n = 0;
	for (int k = 0; k < p_in.n; ++k) {
		const int k1 = (k + 1) % p_in.n;
		const double fp = p_a * p_in.x[k] + p_b * p_in.y[k] + p_c;
		const double fq = p_a * p_in.x[k1] + p_b * p_in.y[k1] + p_c;
		if (fp > 0.0) {
			r_out.push(p_in.x[k], p_in.y[k]);
		}
		if ((fp > 0.0) != (fq > 0.0)) {
			const double t = fp / (fp - fq);
			r_out.push(p_in.x[k] + t * (p_in.x[k1] - p_in.x[k]), p_in.y[k] + t * (p_in.y[k1] - p_in.y[k]));
		}
	}
}

// p_poly clipped by the convex p_clip (both made counter-clockwise first)
void clip_convex(const Poly &p_poly, const Poly &p_clip, Poly &r_out) {
	Poly cur = p_poly;
	make_ccw(cur);
	Poly clip = p_clip;
	make_ccw(clip);
	for (int k = 0; k < clip.n; ++k) {
		if (cur.n == 0) {
			break;
		}
		const int k1 = (k + 1) % clip.n;
		const double a = -(clip.y[k1] - clip.y[k]);
		const double b = clip.x[k1] - clip.x[k];
		const double c = -(a * clip.x[k] + b * clip.y[k]);
		Poly next;
		clip_halfplane(cur, a, b, c, next);
		cur = next;
	}
	r_out = cur;
}

// p_ccw must be counter-clockwise
bool inside_convex(const Poly &p_ccw, double p_x, double p_y) {
	for (int k = 0; k < p_ccw.n; ++k) {
		const int k1 = (k + 1) % p_ccw.n;
		if ((p_ccw.x[k1] - p_ccw.x[k]) * (p_y - p_ccw.y[k]) - (p_ccw.y[k1] - p_ccw.y[k]) * (p_x - p_ccw.x[k]) < -1e-12) {
			return false;
		}
	}
	return true;
}

void poly_bounds(const Poly &p_poly, double &r_x0, double &r_x1, double &r_y0, double &r_y1) {
	r_x0 = r_x1 = p_poly.x[0];
	r_y0 = r_y1 = p_poly.y[0];
	for (int k = 1; k < p_poly.n; ++k) {
		r_x0 = MIN(r_x0, p_poly.x[k]);
		r_x1 = MAX(r_x1, p_poly.x[k]);
		r_y0 = MIN(r_y0, p_poly.y[k]);
		r_y1 = MAX(r_y1, p_poly.y[k]);
	}
}

Poly rect(double p_x0, double p_x1, double p_y0, double p_y1) {
	Poly r;
	r.push(p_x0, p_y0);
	r.push(p_x1, p_y0);
	r.push(p_x1, p_y1);
	r.push(p_x0, p_y1);
	return r;
}

// p_ccw must be counter-clockwise; true when the point is inside or within p_tol of its edges
bool inside_convex_tol(const Poly &p_ccw, double p_x, double p_y, double p_tol) {
	for (int k = 0; k < p_ccw.n; ++k) {
		const int k1 = (k + 1) % p_ccw.n;
		const double ex = p_ccw.x[k1] - p_ccw.x[k];
		const double ey = p_ccw.y[k1] - p_ccw.y[k];
		const double len = std::sqrt(ex * ex + ey * ey);
		if (len > 0.0 && (ex * (p_y - p_ccw.y[k]) - ey * (p_x - p_ccw.x[k])) / len < -p_tol) {
			return false;
		}
	}
	return true;
}

// The partner's faces a soft face is pressed by (counter-clockwise, in the patch frame): one, or several coplanar
// ones (brief 5.2), with each face's world centroid, the fixed order in which they are taken
constexpr int MAX_PARTNERS = 8;

struct PartnerSet {
	int n = 0;
	Poly face[MAX_PARTNERS];
	double key[MAX_PARTNERS][3];
};

bool inside_any(const PartnerSet &p_set, double p_x, double p_y) {
	for (int i = 0; i < p_set.n; ++i) {
		if (inside_convex(p_set.face[i], p_x, p_y)) {
			return true;
		}
	}
	return false;
}

// The exact area moments of a counter-clockwise polygon: A, the integrals of x and y, and of x^2, y^2 and x y
struct PolyMoments {
	double A = 0.0, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
};

PolyMoments poly_moments(const Poly &p_ccw) {
	PolyMoments m;
	for (int k = 0; k < p_ccw.n; ++k) {
		const int k1 = (k + 1) % p_ccw.n;
		const double x0 = p_ccw.x[k], y0 = p_ccw.y[k], x1 = p_ccw.x[k1], y1 = p_ccw.y[k1];
		const double c = x0 * y1 - x1 * y0;
		m.A += c;
		m.Sx += (x0 + x1) * c;
		m.Sy += (y0 + y1) * c;
		m.Sxx += (x0 * x0 + x0 * x1 + x1 * x1) * c;
		m.Syy += (y0 * y0 + y0 * y1 + y1 * y1) * c;
		m.Sxy += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * c;
	}
	m.A /= 2.0;
	m.Sx /= 6.0;
	m.Sy /= 6.0;
	m.Sxx /= 12.0;
	m.Syy /= 12.0;
	m.Sxy /= 24.0;
	return m;
}

// Seven-point quadrature on a triangle, exact to degree 5 (barycentric weights a, b, c and the point's weight)
const double TRI_QA[7] = { 1.0 / 3.0, 0.059715871789770, 0.470142064105115, 0.470142064105115, 0.797426985353087, 0.101286507323456, 0.101286507323456 };
const double TRI_QB[7] = { 1.0 / 3.0, 0.470142064105115, 0.059715871789770, 0.470142064105115, 0.101286507323456, 0.797426985353087, 0.101286507323456 };
const double TRI_QW[7] = { 0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827 };

// ---------------------------------------------------------------------------------------------------------------
// The face grid (design 4.2, brief 3.1): cell-centred over a box in the patch frame; arrays are indexed
// i_long * n_short + i_short, so the half-bandwidth is n_short.

struct Grid {
	int nl = 0;
	int ns = 0;
	double dl = 0.0;
	double ds = 0.0;
	bool long_is_x = true;
	double x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0;

	int n() const { return nl * ns; }
	double area() const { return dl * ds; }
	double X(int p_k) const { return long_is_x ? x0 + (double(p_k / ns) + 0.5) * dl : x0 + (double(p_k % ns) + 0.5) * ds; }
	double Y(int p_k) const { return long_is_x ? y0 + (double(p_k % ns) + 0.5) * ds : y0 + (double(p_k / ns) + 0.5) * dl; }
	double cell_dx() const { return long_is_x ? dl : ds; }
	double cell_dy() const { return long_is_x ? ds : dl; }
};

// n_short across the short side; along the long one 2 clamp(round(4 L / W), 4, 12), even so the half grid is exact
bool make_grid(double p_x0, double p_x1, double p_y0, double p_y1, int p_n_long, int p_n_short, Grid &r_grid) {
	const double lx = p_x1 - p_x0;
	const double ly = p_y1 - p_y0;
	r_grid.long_is_x = lx >= ly;
	const double L = r_grid.long_is_x ? lx : ly;
	const double W = r_grid.long_is_x ? ly : lx;
	if (!(W > 0.0) || !(L > 0.0) || !std::isfinite(L / W)) {
		return false;
	}
	int n_long = p_n_long;
	if (n_long <= 0) {
		// round half to even, as the reference does
		n_long = 2 * int(MIN(MAX(std::nearbyint(4.0 * L / W), 4.0), double(N_LONG_MAX / 2)));
	}
	r_grid.nl = n_long;
	r_grid.ns = p_n_short;
	r_grid.dl = L / n_long;
	r_grid.ds = W / p_n_short;
	r_grid.x0 = p_x0;
	r_grid.x1 = p_x1;
	r_grid.y0 = p_y0;
	r_grid.y1 = p_y1;
	return r_grid.n() <= MAX_CELLS;
}

struct QSystem {
	double diag[MAX_CELLS];
	double off_l[MAX_CELLS]; // coupling of cell k to k + n_short (the next along the long axis)
	double off_s[MAX_CELLS]; // coupling of cell k to k + 1 (the next across)
	double rhs[MAX_CELLS];
	double band[MAX_CELLS * (N_SHORT + 1)]; // the Cholesky factor, row i holding columns i - m .. i
	bool factored = false;
};

// The q system: lam2 q - lap q = kI w, with w = d on active cells (the right-hand side) and w = -(h/E) q on inactive
// domain cells (on the diagonal); q = 0 outside the domain (identity rows) and on the faces bordering it (ghost -q)
void assemble(const Grid &p_grid, const bool *p_dom, const bool *p_active, const Coeffs &p_co, const double *p_d, QSystem &r_sys) {
	const int nl = p_grid.nl;
	const int ns = p_grid.ns;
	const double il2 = 1.0 / (p_grid.dl * p_grid.dl);
	const double is2 = 1.0 / (p_grid.ds * p_grid.ds);
	for (int k = 0; k < p_grid.n(); ++k) {
		const int i = k / ns;
		const int j = k % ns;
		const bool dom = p_dom[k];
		const bool lp = i + 1 < nl && p_dom[k + ns];
		const bool lm = i > 0 && p_dom[k - ns];
		const bool sp = j + 1 < ns && p_dom[k + 1];
		const bool sm = j > 0 && p_dom[k - 1];
		double diag = dom ? p_co.lam2 + 2.0 * il2 + 2.0 * is2 : 1.0;
		diag += (dom && !lp ? il2 : 0.0) + (dom && !lm ? il2 : 0.0);
		diag += (dom && !sp ? is2 : 0.0) + (dom && !sm ? is2 : 0.0);
		diag += (dom && !p_active[k]) ? p_co.kI / p_co.Eh : 0.0;
		r_sys.diag[k] = diag;
		r_sys.off_l[k] = (dom && lp) ? -il2 : 0.0;
		r_sys.off_s[k] = (dom && sp) ? -is2 : 0.0;
		r_sys.rhs[k] = p_active[k] ? p_co.kI * p_d[k] : 0.0;
	}
	r_sys.factored = false;
}

// The banded Cholesky factor L L^T (half-bandwidth m = n_short): fixed sizes, fixed order, no pivoting
bool factor(const Grid &p_grid, QSystem &r_sys) {
	const int n = p_grid.n();
	const int m = p_grid.ns;
	auto A = [&](int p_i, int p_j) -> double {
		// p_i >= p_j
		if (p_i == p_j) {
			return r_sys.diag[p_i];
		}
		if (p_i - p_j == 1 && m != 1) {
			return r_sys.off_s[p_j];
		}
		if (p_i - p_j == m) {
			return r_sys.off_l[p_j];
		}
		return 0.0;
	};
	auto L = [&](int p_i, int p_j) -> double & { return r_sys.band[p_i * (m + 1) + (p_j - p_i + m)]; };
	for (int j = 0; j < n; ++j) {
		const int lo = MAX(0, j - m);
		double s = A(j, j);
		for (int k = lo; k < j; ++k) {
			s -= L(j, k) * L(j, k);
		}
		if (!(s > 0.0)) {
			return false;
		}
		L(j, j) = std::sqrt(s);
		const double inv = 1.0 / L(j, j);
		const int hi = MIN(n, j + m + 1);
		for (int i = j + 1; i < hi; ++i) {
			const int lo_i = MAX(0, i - m);
			double v = A(i, j);
			for (int k = lo_i; k < j; ++k) {
				v -= L(i, k) * L(j, k);
			}
			L(i, j) = v * inv;
		}
	}
	r_sys.factored = true;
	return true;
}

void back_solve(const Grid &p_grid, const QSystem &p_sys, const double *p_b, double *r_x) {
	const int n = p_grid.n();
	const int m = p_grid.ns;
	auto L = [&](int p_i, int p_j) -> double { return p_sys.band[p_i * (m + 1) + (p_j - p_i + m)]; };
	double y[MAX_CELLS];
	for (int i = 0; i < n; ++i) {
		const int lo = MAX(0, i - m);
		double s = p_b[i];
		for (int k = lo; k < i; ++k) {
			s -= L(i, k) * y[k];
		}
		y[i] = s / L(i, i);
	}
	for (int i = n - 1; i >= 0; --i) {
		const int hi = MIN(n, i + m + 1);
		double s = y[i];
		for (int k = i + 1; k < hi; ++k) {
			s -= L(k, i) * r_x[k];
		}
		r_x[i] = s / L(i, i);
	}
}

// The figures of a solved face field (design 4.1 item 4)
struct FaceFig {
	bool empty = true;
	double F = 0.0, cx = 0.0, cy = 0.0, r_s = 0.0, m1x = 0.0, m1y = 0.0, area = 0.0, Ixx = 0.0, Iyy = 0.0, J_p = 0.0;
	double k_n = 0.0, kt_x = 0.0, kt_y = 0.0, kt_xy = 0.0;
	double edge_x[2] = { 0.0, 0.0 }, edge_y[2] = { 0.0, 0.0 };
	int passes = 0;
	bool settled = true;
	Grid grid;
	bool active[MAX_CELLS];
};

// The split law with the contact set by complementarity (p >= 0, w >= d, p (w - d) = 0) on the grid: active-set passes
// from p_active0 (the tail's set) or the geometric overlap, at most ACTIVE_SET_CAP; then the figures, and the normal
// and tilt stiffnesses with the last set held
bool face_grid(const Poly &p_soft_ccw, const PartnerSet &p_partners, double p_d0, double p_gx, double p_gy, const Coeffs &p_co, double p_x0, double p_x1, double p_y0, double p_y1, int p_n_long, int p_n_short, const bool *p_active0, FaceFig &r_fig) {
	Grid &g = r_fig.grid;
	r_fig.empty = true;
	if (!make_grid(p_x0, p_x1, p_y0, p_y1, p_n_long, p_n_short, g)) {
		return false;
	}
	const int n = g.n();
	bool dom[MAX_CELLS], cont[MAX_CELLS];
	double d[MAX_CELLS], X[MAX_CELLS], Y[MAX_CELLS];
	bool *active = r_fig.active;
	for (int k = 0; k < n; ++k) {
		X[k] = g.X(k);
		Y[k] = g.Y(k);
		dom[k] = inside_convex(p_soft_ccw, X[k], Y[k]);
		cont[k] = dom[k] && inside_any(p_partners, X[k], Y[k]);
		d[k] = p_d0 + p_gx * X[k] + p_gy * Y[k];
		active[k] = p_active0 != nullptr ? (p_active0[k] && cont[k] && dom[k]) : (cont[k] && d[k] > 0.0 && dom[k]);
	}

	thread_local QSystem scratch_system;
	QSystem *sys = &scratch_system;
	double q[MAX_CELLS], p[MAX_CELLS];
	int passes = 0;
	bool settled = false;
	bool ok = true;
	for (int it = 0; it < ACTIVE_SET_CAP; ++it) {
		passes = it + 1;
		assemble(g, dom, active, p_co, d, *sys);
		if (!factor(g, *sys)) {
			ok = false;
			break;
		}
		back_solve(g, *sys, sys->rhs, q);
		bool changed = false;
		bool next[MAX_CELLS];
		for (int k = 0; k < n; ++k) {
			p[k] = active[k] ? p_co.Eh * d[k] + q[k] : 0.0;
			const double w = active[k] ? d[k] : -q[k] / p_co.Eh;
			next[k] = active[k];
			if (active[k] && p[k] < 0.0) {
				next[k] = false;
			}
			if (!active[k] && cont[k] && dom[k] && w < d[k]) {
				next[k] = true;
			}
			changed = changed || next[k] != active[k];
		}
		if (!changed) {
			settled = true;
			break;
		}
		for (int k = 0; k < n; ++k) {
			active[k] = next[k];
		}
	}
	r_fig.passes = passes;
	r_fig.settled = settled;
	if (!ok) {
		return false;
	}

	const double A = g.area();
	bool any = false;
	double F = 0.0;
	for (int k = 0; k < n; ++k) {
		F += p[k];
		any = any || active[k];
	}
	F *= A;
	r_fig.F = F;
	if (!(F > 0.0) || !any) {
		return true;
	}
	double sx = 0.0, sy = 0.0;
	for (int k = 0; k < n; ++k) {
		sx += p[k] * X[k];
		sy += p[k] * Y[k];
	}
	const double cx = sx * A / F;
	const double cy = sy * A / F;
	const double dx = g.cell_dx();
	const double dy = g.cell_dy();
	double rs = 0.0, m1x = 0.0, m1y = 0.0, n_act = 0.0, iyy = 0.0, ixx = 0.0, pmax = 0.0;
	double ex0 = 0.0, ex1 = 0.0, ey0 = 0.0, ey1 = 0.0;
	bool first = true;
	for (int k = 0; k < n; ++k) {
		rs += p[k] * std::sqrt((X[k] - cx) * (X[k] - cx) + (Y[k] - cy) * (Y[k] - cy));
		m1x += p[k] * std::abs(X[k] - cx);
		m1y += p[k] * std::abs(Y[k] - cy);
		pmax = MAX(pmax, p[k]);
		if (active[k]) {
			n_act += 1.0;
			iyy += (X[k] - cx) * (X[k] - cx) + dx * dx / 12.0;
			ixx += (Y[k] - cy) * (Y[k] - cy) + dy * dy / 12.0;
			if (first) {
				ex0 = X[k] - dx / 2.0;
				ex1 = X[k] + dx / 2.0;
				ey0 = Y[k] - dy / 2.0;
				ey1 = Y[k] + dy / 2.0;
				first = false;
			} else {
				ex0 = MIN(ex0, X[k] - dx / 2.0);
				ex1 = MAX(ex1, X[k] + dx / 2.0);
				ey0 = MIN(ey0, Y[k] - dy / 2.0);
				ey1 = MAX(ey1, Y[k] + dy / 2.0);
			}
		}
	}
	r_fig.empty = false;
	r_fig.cx = cx;
	r_fig.cy = cy;
	r_fig.r_s = rs * A / F;
	r_fig.m1x = m1x * A / F;
	r_fig.m1y = m1y * A / F;
	r_fig.area = n_act * A;
	r_fig.Iyy = iyy * A;
	r_fig.Ixx = ixx * A;
	r_fig.J_p = r_fig.Ixx + r_fig.Iyy;
	r_fig.edge_x[0] = ex0 - cx;
	r_fig.edge_x[1] = ex1 - cx;
	r_fig.edge_y[0] = ey0 - cy;
	r_fig.edge_y[1] = ey1 - cy;

	// The normal and tilt stiffnesses with the contact set held
	double b1[MAX_CELLS], bx[MAX_CELLS], by[MAX_CELLS], q1[MAX_CELLS], qx[MAX_CELLS], qy[MAX_CELLS];
	for (int k = 0; k < n; ++k) {
		b1[k] = active[k] ? p_co.kI * 1.0 : 0.0;
		bx[k] = active[k] ? p_co.kI * (X[k] - cx) : 0.0;
		by[k] = active[k] ? p_co.kI * (Y[k] - cy) : 0.0;
	}
	back_solve(g, *sys, b1, q1);
	back_solve(g, *sys, bx, qx);
	back_solve(g, *sys, by, qy);
	double kn = 0.0, kty = 0.0, ktx = 0.0, ktxy = 0.0;
	for (int k = 0; k < n; ++k) {
		const double p1 = active[k] ? p_co.Eh + q1[k] : 0.0;
		const double px = active[k] ? p_co.Eh * (X[k] - cx) + qx[k] : 0.0;
		const double py = active[k] ? p_co.Eh * (Y[k] - cy) + qy[k] : 0.0;
		kn += p1;
		kty += px * (X[k] - cx);
		ktx += py * (Y[k] - cy);
		ktxy += px * (Y[k] - cy);
	}
	r_fig.k_n = kn * A;
	r_fig.kt_y = kty * A;
	r_fig.kt_x = ktx * A;
	r_fig.kt_xy = ktxy * A;
	return true;
}

// The distance from the centre to the contact set's edge along an in-plane direction (cos, sin) of the full grid
double edge_distance(const FaceFig &p_fig, double p_mx, double p_my) {
	const Grid &g = p_fig.grid;
	const double hx = 0.5 * g.cell_dx() * std::abs(p_mx);
	const double hy = 0.5 * g.cell_dy() * std::abs(p_my);
	double best = 0.0;
	for (int k = 0; k < g.n(); ++k) {
		if (p_fig.active[k]) {
			best = MAX(best, (g.X(k) - p_fig.cx) * p_mx + (g.Y(k) - p_fig.cy) * p_my + hx + hy);
		}
	}
	return best;
}

// The engine's face law (brief 5.2): the grid box is the overlap's bounding box grown by one layer depth (to the
// layer's own free faces where they are nearer); Richardson on the half grid. p_full keeps the full grid (its
// contact set, edges and arms); r_rich has the Richardson-combined figures.
struct FaceLaw {
	FaceFig full;
	double F = 0.0, cx = 0.0, cy = 0.0, r_s = 0.0, m1x = 0.0, m1y = 0.0, area = 0.0, Ixx = 0.0, Iyy = 0.0, J_p = 0.0;
	double k_n = 0.0, kt_x = 0.0, kt_y = 0.0, kt_xy = 0.0;
	bool empty = true;
	bool richardson = false;
};

// p_partners' faces are counter-clockwise; with several (coplanar faces, brief 5.2) the field is one over all of them
bool face_law(const Poly &p_soft, const PartnerSet &p_partners, double p_d0, double p_gx, double p_gy, const Coeffs &p_co, const JPH::SoftContactTail *p_tail, FaceLaw &r_law) {
	r_law.empty = true;
	// The pressed part of the soft face's overlap with each partner face, and their joint bounds
	bool pressed = false;
	double ox0 = 0.0, ox1 = 0.0, oy0 = 0.0, oy1 = 0.0;
	for (int i = 0; i < p_partners.n; ++i) {
		Poly overlap;
		clip_convex(p_soft, p_partners.face[i], overlap);
		if (overlap.n > 0) {
			Poly cut;
			clip_halfplane(overlap, p_gx, p_gy, p_d0, cut);
			overlap = cut;
		}
		if (overlap.n < 3) {
			continue;
		}
		double a0, a1, b0, b1;
		poly_bounds(overlap, a0, a1, b0, b1);
		if (!pressed) {
			ox0 = a0;
			ox1 = a1;
			oy0 = b0;
			oy1 = b1;
			pressed = true;
		} else {
			ox0 = MIN(ox0, a0);
			ox1 = MAX(ox1, a1);
			oy0 = MIN(oy0, b0);
			oy1 = MAX(oy1, b1);
		}
	}
	if (!pressed) {
		return false;
	}
	double sx0, sx1, sy0, sy1;
	poly_bounds(p_soft, sx0, sx1, sy0, sy1);
	const double g = p_co.h;
	const double bx0 = MAX(ox0 - g, sx0), bx1 = MIN(ox1 + g, sx1), by0 = MAX(oy0 - g, sy0), by1 = MIN(oy1 + g, sy1);

	Poly soft_ccw = p_soft;
	make_ccw(soft_ccw);

	// The tail's contact set when it belongs to the same grid
	bool tail_active[MAX_CELLS];
	const bool *active0 = nullptr;
	{
		Grid probe;
		if (p_tail != nullptr && make_grid(bx0, bx1, by0, by1, 0, N_SHORT, probe) && p_tail->mGridLong == probe.nl && p_tail->mGridShort == probe.ns) {
			for (int k = 0; k < probe.n(); ++k) {
				tail_active[k] = ((p_tail->mActiveSet[k >> 5] >> (k & 31)) & 1u) != 0;
			}
			active0 = tail_active;
		}
	}

	FaceFig &full = r_law.full;
	if (!face_grid(soft_ccw, p_partners, p_d0, p_gx, p_gy, p_co, bx0, bx1, by0, by1, 0, N_SHORT, active0, full) || full.empty) {
		return false;
	}
	r_law.empty = false;
	r_law.F = full.F;
	r_law.cx = full.cx;
	r_law.cy = full.cy;
	r_law.r_s = full.r_s;
	r_law.m1x = full.m1x;
	r_law.m1y = full.m1y;
	r_law.area = full.area;
	r_law.Ixx = full.Ixx;
	r_law.Iyy = full.Iyy;
	r_law.J_p = full.J_p;
	r_law.k_n = full.k_n;
	r_law.kt_x = full.kt_x;
	r_law.kt_y = full.kt_y;
	r_law.kt_xy = full.kt_xy;

	thread_local FaceFig scratch_half;
	FaceFig *half = &scratch_half;
	if (face_grid(soft_ccw, p_partners, p_d0, p_gx, p_gy, p_co, bx0, bx1, by0, by1, full.grid.nl / 2, N_SHORT / 2, nullptr, *half) && !half->empty && half->F > 0.0) {
		auto rich = [](double p_full, double p_half) { return (4.0 * p_full - p_half) / 3.0; };
		r_law.F = rich(full.F, half->F);
		r_law.cx = rich(full.cx, half->cx);
		r_law.cy = rich(full.cy, half->cy);
		r_law.r_s = rich(full.r_s, half->r_s);
		r_law.m1x = rich(full.m1x, half->m1x);
		r_law.m1y = rich(full.m1y, half->m1y);
		r_law.area = rich(full.area, half->area);
		r_law.Ixx = rich(full.Ixx, half->Ixx);
		r_law.Iyy = rich(full.Iyy, half->Iyy);
		r_law.J_p = rich(full.J_p, half->J_p);
		r_law.k_n = rich(full.k_n, half->k_n);
		r_law.kt_x = rich(full.kt_x, half->kt_x);
		r_law.kt_y = rich(full.kt_y, half->kt_y);
		r_law.kt_xy = rich(full.kt_xy, half->kt_xy);
		r_law.richardson = true;
	}
	return true;
}

// TABLE_FORMAT.md 6: S = A / (P h), P from the equivalent rectangle of the contact's second moments
double shape_factor(double p_area, double p_Ixx, double p_Iyy, double p_h) {
	if (!(p_area > 0.0)) {
		return 0.0;
	}
	const double a = std::sqrt(12.0 * p_Iyy / p_area);
	const double b = std::sqrt(12.0 * p_Ixx / p_area);
	return p_area / (2.0 * (a + b) * p_h);
}

// ---------------------------------------------------------------------------------------------------------------
// The layer table (TABLE_FORMAT.v2 4): cubic Hermite (or linear, by INTERP_ORDER) in the log axes, linear in the face
// coordinate, the positive scalars interpolated on their logarithms

struct AxisWeights {
	int count = 0;
	int index[4] = { 0, 0, 0, 0 };
	double w[4] = { 0.0, 0.0, 0.0, 0.0 };
	double dw[4] = { 0.0, 0.0, 0.0, 0.0 };
	int cell = 0;

	void add(int p_index, double p_w, double p_dw) {
		for (int k = 0; k < count; ++k) {
			if (index[k] == p_index) {
				w[k] += p_w;
				dw[k] += p_dw;
				return;
			}
		}
		if (count < 4) {
			index[count] = p_index;
			w[count] = p_w;
			dw[count] = p_dw;
			++count;
		}
	}
};

// d f / d x at node j of the parabola through three neighbouring nodes (one-sided at the ends), as node weights
void deriv_coeffs(const double *p_x, int p_n, int p_j, int r_idx[3], double r_c[3]) {
	if (p_j == 0) {
		const double h1 = p_x[1] - p_x[0], h2 = p_x[2] - p_x[1];
		r_idx[0] = 0;
		r_idx[1] = 1;
		r_idx[2] = 2;
		r_c[0] = -(2 * h1 + h2) / (h1 * (h1 + h2));
		r_c[1] = (h1 + h2) / (h1 * h2);
		r_c[2] = -h1 / (h2 * (h1 + h2));
		return;
	}
	if (p_j == p_n - 1) {
		const double h1 = p_x[p_n - 2] - p_x[p_n - 3], h2 = p_x[p_n - 1] - p_x[p_n - 2];
		r_idx[0] = p_n - 3;
		r_idx[1] = p_n - 2;
		r_idx[2] = p_n - 1;
		r_c[0] = h2 / (h1 * (h1 + h2));
		r_c[1] = -(h1 + h2) / (h1 * h2);
		r_c[2] = (h1 + 2 * h2) / (h2 * (h1 + h2));
		return;
	}
	const double hl = p_x[p_j] - p_x[p_j - 1], hr = p_x[p_j + 1] - p_x[p_j];
	r_idx[0] = p_j - 1;
	r_idx[1] = p_j;
	r_idx[2] = p_j + 1;
	r_c[0] = -hr / (hl * (hl + hr));
	r_c[1] = (hr - hl) / (hl * hr);
	r_c[2] = hl / (hr * (hl + hr));
}

// Weights on the nodes of one log axis at p_value and their derivatives with respect to its logarithm (4.2)
AxisWeights axis_weights(const float *p_axis, int p_n, double p_value, int p_order) {
	double x[16];
	for (int k = 0; k < p_n && k < 16; ++k) {
		x[k] = std::log(double(p_axis[k]));
	}
	const double X = std::log(p_value);
	int i = -1;
	for (int k = 0; k < p_n; ++k) {
		if (x[k] <= X) {
			i = k;
		}
	}
	i = MIN(MAX(i, 0), p_n - 2);
	const double D = x[i + 1] - x[i];
	const double t = (X - x[i]) / D;
	AxisWeights r;
	r.cell = i;
	if (p_order == 1 || p_n < 3) {
		r.add(i, 1.0 - t, -1.0 / D);
		r.add(i + 1, t, 1.0 / D);
		return r;
	}
	int idx[3];
	double c[3];
	if (X < x[0]) {
		// below the first node: the first node's tangent, a power law to zero
		deriv_coeffs(x, p_n, 0, idx, c);
		for (int k = 0; k < 3; ++k) {
			r.add(idx[k], (X - x[0]) * c[k], c[k]);
		}
		r.add(0, 1.0, 0.0);
		r.cell = 0;
		return r;
	}
	const double t2 = t * t, t3 = t2 * t;
	const double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
	const double d00 = 6 * t2 - 6 * t, d10 = 3 * t2 - 4 * t + 1, d01 = -6 * t2 + 6 * t, d11 = 3 * t2 - 2 * t;
	r.add(i, h00, d00 / D);
	r.add(i + 1, h01, d01 / D);
	deriv_coeffs(x, p_n, i, idx, c);
	for (int k = 0; k < 3; ++k) {
		r.add(idx[k], h10 * D * c[k], d10 * c[k]);
	}
	deriv_coeffs(x, p_n, i + 1, idx, c);
	for (int k = 0; k < 3; ++k) {
		r.add(idx[k], h11 * D * c[k], d11 * c[k]);
	}
	return r;
}

// The face coordinate (4.3): linear between the four Coulomb columns and the bonded one at BONDED_MU
void face_coordinate(double p_mu, int &r_f, double &r_tf) {
	double m[table::NUM_FACE];
	for (int k = 0; k < table::NUM_FACE; ++k) {
		m[k] = table::FACE_MU[k] >= 0.0f ? double(table::FACE_MU[k]) : double(table::BONDED_MU);
	}
	if (p_mu >= m[table::NUM_FACE - 1]) {
		r_f = table::NUM_FACE - 2;
		r_tf = 1.0;
		return;
	}
	const double mu = MAX(p_mu, 0.0);
	int f = -1;
	for (int k = 0; k < table::NUM_FACE; ++k) {
		if (m[k] <= mu) {
			f = k;
		}
	}
	f = MIN(MAX(f, 0), table::NUM_FACE - 2);
	r_f = f;
	r_tf = (mu - m[f]) / (m[f + 1] - m[f]);
}

// The K / G cell and its clamped linear coordinate (profiles, T3, T4)
void kg_cell(double p_kg, int &r_k, double &r_t) {
	if (table::NUM_KG == 1) {
		r_k = 0;
		r_t = 0.0;
		return;
	}
	const double X = std::log(p_kg);
	int k = -1;
	for (int i = 0; i < table::NUM_KG; ++i) {
		if (std::log(double(table::KG[i])) <= X) {
			k = i;
		}
	}
	k = MIN(MAX(k, 0), table::NUM_KG - 2);
	const double x0 = std::log(double(table::KG[k])), x1 = std::log(double(table::KG[k + 1]));
	r_k = k;
	r_t = MIN(MAX((X - x0) / (x1 - x0), 0.0), 1.0);
}

struct Lookup {
	double force = 0.0; // CYL: P' / (E h); SPH: F / (E h^2)
	double width = 0.0; // CYL: b / h; SPH: a / h
	double m1 = 0.0; // m1 / h
	double spin = 0.0; // SPH: r_s / h
	double slope = 0.0; // d ln F / d ln squash
	double profile[table::NUM_ST];
	int order = 3;
};

// One lookup (4.1 to 4.6). p_cyl: the plane-strain cylinder, else the axisymmetric sphere. False: outside the table.
bool table_lookup(bool p_cyl, double p_rh, double p_sh, double p_mu, double p_kg, Lookup &r_out) {
	if (!(p_sh > 0.0)) {
		return false;
	}
	if (p_rh < table::RH[0] || p_rh > table::RH[table::NUM_RH - 1] || p_sh > table::SH[table::NUM_SH - 1]) {
		return false;
	}
	if (p_kg < double(table::KG[0]) * (1.0 - 1e-9)) {
		return false;
	}
	int f;
	double tf;
	face_coordinate(p_mu, f, tf);
	const int wf_index[2] = { f, f + 1 };
	const double wf[2] = { 1.0 - tf, tf };
	const int(*axis)[table::NUM_SH] = p_cyl ? table::CYL_KG_AXIS : table::SPH_KG_AXIS;
	const int(*status)[table::NUM_FACE][table::NUM_RH][table::NUM_SH] = p_cyl ? table::CYL_STATUS : table::SPH_STATUS;

	auto kg_for = [&](const AxisWeights &p_wu, const AxisWeights &p_wv, double &r_kg) -> bool {
		if (p_kg > double(table::KG[table::NUM_KG - 1]) * (1.0 + 1e-9)) {
			for (int b = 0; b < p_wu.count; ++b) {
				for (int c = 0; c < p_wv.count; ++c) {
					if (p_wu.w[b] != 0.0 && p_wv.w[c] != 0.0 && axis[p_wu.index[b]][p_wv.index[c]]) {
						return false;
					}
				}
			}
			r_kg = double(table::KG[table::NUM_KG - 1]);
			return true;
		}
		r_kg = MIN(MAX(p_kg, double(table::KG[0])), double(table::KG[table::NUM_KG - 1]));
		return true;
	};
	auto kg_weights = [&](double p_kgc, int p_order) {
		if (table::NUM_KG == 1) {
			AxisWeights r;
			r.add(0, 1.0, 0.0);
			return r;
		}
		return axis_weights(table::KG, table::NUM_KG, p_kgc, p_order);
	};
	auto touches_unsolved = [&](const AxisWeights &p_wk, const AxisWeights &p_wu, const AxisWeights &p_wv) -> bool {
		for (int k = 0; k < p_wk.count; ++k) {
			for (int a = 0; a < 2; ++a) {
				for (int b = 0; b < p_wu.count; ++b) {
					for (int c = 0; c < p_wv.count; ++c) {
						if (p_wk.w[k] != 0.0 && wf[a] != 0.0 && p_wu.w[b] != 0.0 && p_wv.w[c] != 0.0 && status[p_wk.index[k]][wf_index[a]][p_wu.index[b]][p_wv.index[c]]) {
							return true;
						}
					}
				}
			}
		}
		return false;
	};

	int order = table::INTERP_ORDER;
	AxisWeights wu = axis_weights(table::RH, table::NUM_RH, p_rh, order);
	AxisWeights wv = axis_weights(table::SH, table::NUM_SH, p_sh, order);
	double kgc = 0.0;
	if (!kg_for(wu, wv, kgc)) {
		return false;
	}
	AxisWeights wk = kg_weights(kgc, order);
	if (touches_unsolved(wk, wu, wv)) {
		// an unsolved node in the wider stencil drops this lookup to order 1; an unsolved corner of the cell itself is
		// outside the table
		order = 1;
		wu = axis_weights(table::RH, table::NUM_RH, p_rh, 1);
		wv = axis_weights(table::SH, table::NUM_SH, p_sh, 1);
		if (!kg_for(wu, wv, kgc)) {
			return false;
		}
		wk = kg_weights(kgc, 1);
		if (touches_unsolved(wk, wu, wv)) {
			return false;
		}
	}
	r_out.order = order;

	auto scalar = [&](const float (*p_arr)[table::NUM_FACE][table::NUM_RH][table::NUM_SH], bool p_slope, double *r_slope) {
		double acc = 0.0, dacc = 0.0;
		for (int k = 0; k < wk.count; ++k) {
			for (int a = 0; a < 2; ++a) {
				for (int b = 0; b < wu.count; ++b) {
					for (int c = 0; c < wv.count; ++c) {
						const double lq = std::log(double(p_arr[wk.index[k]][wf_index[a]][wu.index[b]][wv.index[c]]));
						acc += wk.w[k] * wf[a] * wu.w[b] * wv.w[c] * lq;
						if (p_slope) {
							dacc += wk.w[k] * wf[a] * wu.w[b] * wv.dw[c] * lq;
						}
					}
				}
			}
		}
		if (r_slope != nullptr) {
			*r_slope = dacc;
		}
		return std::exp(acc);
	};
	if (p_cyl) {
		r_out.force = scalar(table::CYL_FORCE, true, &r_out.slope);
		r_out.width = scalar(table::CYL_HALFWIDTH, false, nullptr);
		r_out.m1 = scalar(table::CYL_M1, false, nullptr);
		r_out.spin = 0.0;
	} else {
		r_out.force = scalar(table::SPH_FORCE, true, &r_out.slope);
		r_out.width = scalar(table::SPH_RADIUS, false, nullptr);
		r_out.m1 = scalar(table::SPH_M1, false, nullptr);
		r_out.spin = scalar(table::SPH_SPIN, false, nullptr);
	}

	// Profiles: bilinear in the two logs with both cell coordinates clamped, linear in the face and in ln K/G
	const int i = wu.cell;
	const int j = wv.cell;
	double lu = (std::log(p_rh) - std::log(double(table::RH[i]))) / (std::log(double(table::RH[i + 1])) - std::log(double(table::RH[i])));
	double lv = (std::log(p_sh) - std::log(double(table::SH[j]))) / (std::log(double(table::SH[j + 1])) - std::log(double(table::SH[j])));
	lu = MIN(MAX(lu, 0.0), 1.0);
	lv = MIN(MAX(lv, 0.0), 1.0);
	int kc;
	double lk;
	kg_cell(kgc, kc, lk);
	const int npairs = table::NUM_KG == 1 ? 1 : 2;
	const int kidx[2] = { kc, MIN(kc + 1, table::NUM_KG - 1) };
	const double kw[2] = { table::NUM_KG == 1 ? 1.0 : 1.0 - lk, lk };
	for (int s = 0; s < table::NUM_ST; ++s) {
		r_out.profile[s] = 0.0;
	}
	for (int kk = 0; kk < npairs; ++kk) {
		for (int a = 0; a < 2; ++a) {
			for (int bb = 0; bb < 2; ++bb) {
				for (int cc = 0; cc < 2; ++cc) {
					const double w = kw[kk] * wf[a] * (bb == 0 ? 1.0 - lu : lu) * (cc == 0 ? 1.0 - lv : lv);
					for (int s = 0; s < table::NUM_ST; ++s) {
						const float v = p_cyl ? table::CYL_PROFILE[kidx[kk]][wf_index[a]][i + bb][j + cc][s] : table::SPH_PROFILE[kidx[kk]][wf_index[a]][i + bb][j + cc][s];
						r_out.profile[s] += w * double(v);
					}
				}
			}
		}
	}
	return true;
}

// T3 (p_t4 false, x = S) or T4 (x = c/h) (4.7): linear in ln x on the ratio, the face coordinate, linear in ln K/G with
// K/G clamped to the axis
double table_ratio(bool p_t4, double p_x, double p_mu, double p_kg) {
	const float *axis = p_t4 ? table::T4_C : table::T3_S;
	const int n = p_t4 ? table::T4_NUM_C : table::T3_NUM_S;
	const double x = MIN(MAX(p_x, double(axis[0])), double(axis[n - 1]));
	const AxisWeights w = axis_weights(axis, n, x, 1);
	int f;
	double tf;
	face_coordinate(p_mu, f, tf);
	const double kg = MIN(MAX(p_kg, double(table::KG[0])), double(table::KG[table::NUM_KG - 1]));
	int kc;
	double lk;
	kg_cell(kg, kc, lk);
	const int npairs = table::NUM_KG == 1 ? 1 : 2;
	const int kidx[2] = { kc, MIN(kc + 1, table::NUM_KG - 1) };
	const double kw[2] = { table::NUM_KG == 1 ? 1.0 : 1.0 - lk, lk };
	double r = 0.0;
	for (int kk = 0; kk < npairs; ++kk) {
		for (int a = 0; a < 2; ++a) {
			const double wa = a == 0 ? 1.0 - tf : tf;
			for (int b = 0; b < w.count; ++b) {
				const float v = p_t4 ? table::T4_RATIO[kidx[kk]][f + a][w.index[b]] : table::T3_RATIO[kidx[kk]][f + a][w.index[b]];
				r += kw[kk] * wa * w.w[b] * double(v);
			}
		}
	}
	return r;
}

double kg_of(const JoltSoftSurface3D &p_s) {
	double G, nu;
	elastic(p_s.modulus, p_s.bulk_modulus, G, nu);
	return p_s.bulk_modulus / G;
}

// ---------------------------------------------------------------------------------------------------------------
// Curved patches: the table's strip model along a Line and its axisymmetric figures at a Point

struct CurvedFig {
	bool empty = true;
	double F = 0.0, cx = 0.0, cy = 0.0, r_s = 0.0, m1x = 0.0, m1y = 0.0, k_n = 0.0, kt_x = 0.0, area = 0.0, J_p = 0.0;
	double sigma = 0.0, n_exp = 1.0;
	double Ixx = 0.0, Iyy = 0.0; // the solved contact's second moments (outside the table: T3's shape factor)
	double y_lo = 0.0, y_hi = 0.0; // the loaded ends along the line
	double face_ratio = 1.0;
	bool outside_table = false;
};

// Simpson weights over theta = pi k / 16, k = 0 .. 8 (TABLE_FORMAT 3)
const double SIMPSON_W[9] = { 1, 4, 2, 4, 2, 4, 2, 4, 1 };

bool table_line(double p_s0, double p_camber, double p_ya, double p_yb, double p_R, const JoltSoftSurface3D &p_layer, double p_mu, CurvedFig &r_fig) {
	const double h = p_layer.thickness;
	const double E = p_layer.modulus;
	const double kg = kg_of(p_layer);
	const double dy = (p_yb - p_ya) / N_SLICES;
	double ys[N_SLICES], P[N_SLICES], b[N_SLICES], m1[N_SLICES], dP[N_SLICES], prof[N_SLICES][table::NUM_ST];
	for (int i = 0; i < N_SLICES; ++i) {
		ys[i] = p_ya + (double(i) + 0.5) * dy;
		P[i] = b[i] = m1[i] = dP[i] = 0.0;
		const double s = p_s0 + p_camber * ys[i];
		if (!(s > 0.0)) {
			continue;
		}
		Lookup r;
		if (!table_lookup(true, p_R / h, s / h, p_mu, kg, r)) {
			return false;
		}
		P[i] = E * h * r.force;
		b[i] = h * r.width;
		m1[i] = h * r.m1;
		dP[i] = P[i] * r.slope / s;
		for (int k = 0; k < table::NUM_ST; ++k) {
			prof[i][k] = r.profile[k];
		}
	}
	double F = 0.0;
	for (int i = 0; i < N_SLICES; ++i) {
		F += P[i];
	}
	F *= dy;
	r_fig.F = F;
	if (!(F > 0.0)) {
		r_fig.empty = true;
		return true;
	}
	double sy = 0.0;
	for (int i = 0; i < N_SLICES; ++i) {
		sy += P[i] * ys[i];
	}
	const double cy = sy * dy / F;
	double rs_num = 0.0;
	bool first = true;
	for (int i = 0; i < N_SLICES; ++i) {
		if (!(P[i] > 0.0)) {
			continue;
		}
		const double pm = P[i] / (2.0 * b[i]);
		double acc = 0.0;
		for (int k = 0; k < table::NUM_ST; ++k) {
			const double theta = PI * k / 16.0;
			const double x = b[i] * std::sin(theta);
			const double dist = std::sqrt(x * x + (ys[i] - cy) * (ys[i] - cy));
			acc += (PI / 48.0) * SIMPSON_W[k] * pm * prof[i][k] * dist * b[i] * std::cos(theta);
		}
		rs_num += 2.0 * dy * acc;
		if (first) {
			r_fig.y_lo = ys[i] - 0.5 * dy;
			first = false;
		}
		r_fig.y_hi = ys[i] + 0.5 * dy;
	}
	double area = 0.0, J_p = 0.0, m1s = 0.0, kn = 0.0, ktx = 0.0, sig = 0.0;
	for (int i = 0; i < N_SLICES; ++i) {
		area += 2.0 * b[i] * dy;
		J_p += 2.0 * b[i] * dy * ((ys[i] - cy) * (ys[i] - cy) + dy * dy / 12.0) + 2.0 * b[i] * dy * (2.0 * b[i]) * (2.0 * b[i]) / 12.0;
		m1s += P[i] * m1[i];
		kn += dP[i];
		ktx += dP[i] * (ys[i] - cy) * (ys[i] - cy);
		sig += P[i] * b[i];
	}
	r_fig.empty = false;
	r_fig.cx = 0.0;
	r_fig.cy = cy;
	r_fig.r_s = rs_num / F;
	r_fig.m1x = m1s * dy / F;
	r_fig.m1y = 0.0;
	r_fig.k_n = kn * dy;
	r_fig.kt_x = ktx * dy;
	r_fig.area = area;
	r_fig.J_p = J_p;
	r_fig.sigma = sig * dy / F;
	const double sc = p_s0 + p_camber * cy;
	r_fig.n_exp = sc > 0.0 ? r_fig.k_n * sc / F : 1.5;
	return true;
}

bool table_point(double p_s, double p_R, const JoltSoftSurface3D &p_layer, double p_mu, CurvedFig &r_fig) {
	const double h = p_layer.thickness;
	const double E = p_layer.modulus;
	Lookup r;
	if (!table_lookup(false, p_R / h, p_s / h, p_mu, kg_of(p_layer), r)) {
		return false;
	}
	const double F = E * h * h * r.force;
	const double a = h * r.width;
	r_fig.empty = !(F > 0.0);
	r_fig.F = F;
	r_fig.cx = r_fig.cy = 0.0;
	r_fig.r_s = h * r.spin;
	r_fig.m1x = r_fig.m1y = h * r.m1;
	r_fig.k_n = F * r.slope / p_s;
	r_fig.area = PI * a * a;
	r_fig.J_p = 0.5 * PI * a * a * a * a;
	r_fig.sigma = a;
	r_fig.n_exp = r.slope;
	r_fig.y_lo = -a;
	r_fig.y_hi = a;
	return true;
}

// Outside the table: the split law with complementarity on the grid, d = s0 + camber y - x^2 / (2R) across a Line
// (the layer unbounded across, ending at the line's ends) and s0 - r^2 / (2R) at a Point
bool curved_outside(bool p_line, double p_s0, double p_R, const Coeffs &p_co, double p_camber, double p_ya, double p_yb, CurvedFig &r_fig) {
	const double h = p_co.h;
	const double smax = p_line ? p_s0 + std::abs(p_camber) * MAX(std::abs(p_ya), std::abs(p_yb)) : p_s0;
	const double bg = std::sqrt(2.0 * p_R * MAX(smax, 1e-12));
	double bx0, bx1, by0, by1;
	if (p_line) {
		bx0 = -(bg + h);
		bx1 = bg + h;
		by0 = p_ya;
		by1 = p_yb;
	} else {
		bx0 = by0 = -(bg + h);
		bx1 = by1 = bg + h;
	}
	Poly soft = rect(bx0, bx1, by0, by1);
	make_ccw(soft);
	const double big = 1e6;
	Poly partner = rect(-big, big, -big, big);
	make_ccw(partner);

	// The field is curved: the grid is solved by the same routine with d per cell
	auto solve = [&](int p_n_long, int p_n_short, FaceFig &r_out) -> bool {
		Grid &g = r_out.grid;
		r_out.empty = true;
		if (!make_grid(bx0, bx1, by0, by1, p_n_long, p_n_short, g)) {
			return false;
		}
		// d = s0 + camber y - x^2 / (2 R) (Line) or s0 - r^2 / (2 R) (Point) is not linear: solve cell by cell through
		// face_grid's machinery with a linear field would be wrong, so this repeats its passes with the curved field
		const int n = g.n();
		bool dom[MAX_CELLS], cont[MAX_CELLS];
		double d[MAX_CELLS], X[MAX_CELLS], Y[MAX_CELLS];
		bool *active = r_out.active;
		for (int k = 0; k < n; ++k) {
			X[k] = g.X(k);
			Y[k] = g.Y(k);
			dom[k] = inside_convex(soft, X[k], Y[k]);
			cont[k] = dom[k] && inside_convex(partner, X[k], Y[k]);
			d[k] = p_line ? p_s0 + p_camber * Y[k] - X[k] * X[k] / (2.0 * p_R) : p_s0 - (X[k] * X[k] + Y[k] * Y[k]) / (2.0 * p_R);
			active[k] = cont[k] && d[k] > 0.0 && dom[k];
		}
		thread_local QSystem scratch_curved_system;
		QSystem *sys = &scratch_curved_system;
		double q[MAX_CELLS], p[MAX_CELLS];
		bool ok = true;
		for (int it = 0; it < ACTIVE_SET_CAP; ++it) {
			assemble(g, dom, active, p_co, d, *sys);
			if (!factor(g, *sys)) {
				ok = false;
				break;
			}
			back_solve(g, *sys, sys->rhs, q);
			bool changed = false;
			bool next[MAX_CELLS];
			for (int k = 0; k < n; ++k) {
				p[k] = active[k] ? p_co.Eh * d[k] + q[k] : 0.0;
				const double w = active[k] ? d[k] : -q[k] / p_co.Eh;
				next[k] = active[k];
				if (active[k] && p[k] < 0.0) {
					next[k] = false;
				}
				if (!active[k] && cont[k] && dom[k] && w < d[k]) {
					next[k] = true;
				}
				changed = changed || next[k] != active[k];
			}
			if (!changed) {
				break;
			}
			for (int k = 0; k < n; ++k) {
				active[k] = next[k];
			}
		}
		if (!ok) {
			return false;
		}
		const double A = g.area();
		double F = 0.0;
		for (int k = 0; k < n; ++k) {
			F += p[k];
		}
		F *= A;
		r_out.F = F;
		if (!(F > 0.0)) {
			return true;
		}
		double sx = 0.0, sy = 0.0;
		for (int k = 0; k < n; ++k) {
			sx += p[k] * X[k];
			sy += p[k] * Y[k];
		}
		const double cx = sx * A / F, cy = sy * A / F;
		const double dx = g.cell_dx(), dy = g.cell_dy();
		double rs = 0.0, m1x = 0.0, m1y = 0.0, n_act = 0.0, iyy = 0.0, ixx = 0.0;
		for (int k = 0; k < n; ++k) {
			rs += p[k] * std::sqrt((X[k] - cx) * (X[k] - cx) + (Y[k] - cy) * (Y[k] - cy));
			m1x += p[k] * std::abs(X[k] - cx);
			m1y += p[k] * std::abs(Y[k] - cy);
			if (active[k]) {
				n_act += 1.0;
				iyy += (X[k] - cx) * (X[k] - cx) + dx * dx / 12.0;
				ixx += (Y[k] - cy) * (Y[k] - cy) + dy * dy / 12.0;
			}
		}
		r_out.empty = false;
		r_out.cx = cx;
		r_out.cy = cy;
		r_out.r_s = rs * A / F;
		r_out.m1x = m1x * A / F;
		r_out.m1y = m1y * A / F;
		r_out.area = n_act * A;
		r_out.Iyy = iyy * A;
		r_out.Ixx = ixx * A;
		r_out.J_p = r_out.Ixx + r_out.Iyy;
		double b1[MAX_CELLS], bx[MAX_CELLS], by[MAX_CELLS], q1[MAX_CELLS], qx[MAX_CELLS], qy[MAX_CELLS];
		for (int k = 0; k < n; ++k) {
			b1[k] = active[k] ? p_co.kI * 1.0 : 0.0;
			bx[k] = active[k] ? p_co.kI * (X[k] - cx) : 0.0;
			by[k] = active[k] ? p_co.kI * (Y[k] - cy) : 0.0;
		}
		back_solve(g, *sys, b1, q1);
		back_solve(g, *sys, bx, qx);
		back_solve(g, *sys, by, qy);
		double kn = 0.0, kty = 0.0, ktx = 0.0, ktxy = 0.0;
		for (int k = 0; k < n; ++k) {
			const double p1 = active[k] ? p_co.Eh + q1[k] : 0.0;
			const double px = active[k] ? p_co.Eh * (X[k] - cx) + qx[k] : 0.0;
			const double py = active[k] ? p_co.Eh * (Y[k] - cy) + qy[k] : 0.0;
			kn += p1;
			kty += px * (X[k] - cx);
			ktx += py * (Y[k] - cy);
			ktxy += px * (Y[k] - cy);
		}
		r_out.k_n = kn * A;
		r_out.kt_y = kty * A;
		r_out.kt_x = ktx * A;
		r_out.kt_xy = ktxy * A;
		return true;
	};

	thread_local FaceFig scratch_curved_full;
	thread_local FaceFig scratch_curved_half;
	FaceFig *full = &scratch_curved_full;
	FaceFig *half = &scratch_curved_half;
	bool ok = solve(0, N_SHORT, *full) && !full->empty;
	if (ok) {
		double F = full->F, cx = full->cx, cy = full->cy, rs = full->r_s, m1x = full->m1x, m1y = full->m1y, area = full->area, Ixx = full->Ixx, Iyy = full->Iyy, kn = full->k_n, ktx = full->kt_x;
		if (solve(full->grid.nl / 2, N_SHORT / 2, *half) && !half->empty && half->F > 0.0) {
			auto rich = [](double p_full, double p_half) { return (4.0 * p_full - p_half) / 3.0; };
			F = rich(F, half->F);
			cx = rich(cx, half->cx);
			cy = rich(cy, half->cy);
			rs = rich(rs, half->r_s);
			m1x = rich(m1x, half->m1x);
			m1y = rich(m1y, half->m1y);
			area = rich(area, half->area);
			Ixx = rich(Ixx, half->Ixx);
			Iyy = rich(Iyy, half->Iyy);
			kn = rich(kn, half->k_n);
			ktx = rich(ktx, half->kt_x);
		}
		r_fig.empty = false;
		r_fig.F = F;
		r_fig.cx = cx;
		r_fig.cy = cy;
		r_fig.r_s = rs;
		r_fig.m1x = m1x;
		r_fig.m1y = m1y;
		r_fig.area = area;
		r_fig.J_p = Ixx + Iyy;
		r_fig.k_n = kn;
		r_fig.kt_x = ktx;
		r_fig.sigma = p_line ? m1x * 8.0 / 3.0 : std::sqrt(MAX(area, 0.0) / PI);
		r_fig.n_exp = p_line ? 1.5 : 2.0;
		r_fig.outside_table = true;
		// T3 on the solved contact (the face condition)
		r_fig.face_ratio = 1.0;
		double ey0 = 0.0, ey1 = 0.0;
		bool first = true;
		for (int k = 0; k < full->grid.n(); ++k) {
			if (full->active[k]) {
				const double y0 = full->grid.Y(k) - 0.5 * full->grid.cell_dy();
				const double y1 = full->grid.Y(k) + 0.5 * full->grid.cell_dy();
				ey0 = first ? y0 : MIN(ey0, y0);
				ey1 = first ? y1 : MAX(ey1, y1);
				first = false;
			}
		}
		r_fig.y_lo = ey0;
		r_fig.y_hi = ey1;
		r_fig.Ixx = Ixx;
		r_fig.Iyy = Iyy;
	}
	return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Solid (design 2.3) and Lumped (design 2.4)

double e_star(const JoltSoftSurface3D &p_s1, const JoltSoftSurface3D *p_s2) {
	double G, nu;
	elastic(p_s1.modulus, p_s1.bulk_modulus, G, nu);
	double inv = (1.0 - nu * nu) / p_s1.modulus;
	if (p_s2 != nullptr && p_s2->kind == JoltSoftSurface3D::KIND_SOLID) {
		double G2, nu2;
		elastic(p_s2->modulus, p_s2->bulk_modulus, G2, nu2);
		inv += (1.0 - nu2 * nu2) / p_s2->modulus;
	}
	return 1.0 / inv;
}

// UNSOURCED (design 2.3, desk E): approach d = (2 P' / (pi E*)) (ln(4R / b) - 1/2), b = sqrt(4 P' R / (pi E*)), inverted
// for P' by bisection in log space (a fixed count)
void solid_line_load(double p_s, double p_R, double p_Es, double &r_P, double &r_b) {
	double lo = 1e-12, hi = 1e9;
	for (int it = 0; it < 60; ++it) {
		const double P = std::sqrt(lo * hi);
		const double b = std::sqrt(4.0 * P * p_R / (PI * p_Es));
		const double d = (2.0 * P / (PI * p_Es)) * (std::log(4.0 * p_R / b) - 0.5);
		if (d > p_s) {
			hi = P;
		} else {
			lo = P;
		}
	}
	r_P = std::sqrt(lo * hi);
	r_b = std::sqrt(4.0 * r_P * p_R / (PI * p_Es));
}

// ---------------------------------------------------------------------------------------------------------------
// Geometry: the leaf under a sub-shape, its scale, and the supporting faces

const JPH::Shape *leaf_shape(const JPH::Shape *p_shape, JPH::SubShapeID p_id, JPH::Vec3 &r_scale) {
	r_scale = JPH::Vec3::sOne();
	const JPH::Shape *shape = p_shape;
	for (int depth = 0; depth < 16 && shape != nullptr; ++depth) {
		if (shape->GetType() == JPH::EShapeType::Compound) {
			const JPH::CompoundShape *compound = static_cast<const JPH::CompoundShape *>(shape);
			JPH::SubShapeID remainder;
			const JPH::uint index = compound->GetSubShapeIndexFromID(p_id, remainder);
			const JPH::CompoundShape::SubShape &sub = compound->GetSubShape(index);
			r_scale = sub.TransformScale(r_scale);
			shape = sub.mShape;
			p_id = remainder;
			continue;
		}
		if (shape->GetType() == JPH::EShapeType::Decorated) {
			if (shape->GetSubType() == JPH::EShapeSubType::Scaled) {
				r_scale = r_scale * static_cast<const JPH::ScaledShape *>(shape)->GetScale();
			} else if (shape->GetSubType() == JPH::EShapeSubType::RotatedTranslated) {
				r_scale = static_cast<const JPH::RotatedTranslatedShape *>(shape)->TransformScale(r_scale);
			}
			shape = static_cast<const JPH::DecoratedShape *>(shape)->GetInnerShape();
			continue;
		}
		return shape;
	}
	return shape;
}

// The radius of curvature a leaf presents across its curved side (0: flat or unknown)
double leaf_radius(const JPH::Shape *p_leaf, const JPH::Vec3 &p_scale) {
	if (p_leaf == nullptr) {
		return 0.0;
	}
	const double s = std::abs(double(p_scale.GetX()));
	switch (p_leaf->GetSubType()) {
		case JPH::EShapeSubType::Sphere:
			return double(static_cast<const JPH::SphereShape *>(p_leaf)->GetRadius()) * s;
		case JPH::EShapeSubType::Cylinder:
			return double(static_cast<const JPH::CylinderShape *>(p_leaf)->GetRadius()) * s;
		case JPH::EShapeSubType::Capsule:
			return double(static_cast<const JPH::CapsuleShape *>(p_leaf)->GetRadius()) * s;
		default:
			return 0.0;
	}
}

struct SupportFace {
	int n = 0;
	JPH::Vec3 v[32]; // relative to the manifold's base offset
};

// Brief 5.2: the partner body's faces that lie in its pressed face's plane (outward normal within 1 deg of
// p_out_normal, plane within the speculative distance of p_partner_face) and overlap the soft face, projected on the
// patch frame (o, t1, t2), counter-clockwise, in a fixed order (by world centroid). Empty for a partner of one convex
// shape, which has one face there.
void coplanar_faces(const JPH::Body &p_partner, JPH::Vec3Arg p_out_normal, const SupportFace &p_partner_face, const SupportFace &p_soft_face, const Poly &p_soft_ccw, JPH::RVec3Arg p_base, const JPH::Vec3 &p_o, const JPH::Vec3 &p_t1, const JPH::Vec3 &p_t2, PartnerSet &r_set) {
	r_set.n = 0;
	const JPH::Shape *root = p_partner.GetShape();
	for (int depth = 0; depth < 16 && root != nullptr && root->GetType() == JPH::EShapeType::Decorated; ++depth) {
		root = static_cast<const JPH::DecoratedShape *>(root)->GetInnerShape();
	}
	if (root == nullptr || root->GetType() == JPH::EShapeType::Convex || p_partner_face.n < 3 || p_soft_face.n < 3) {
		return;
	}
	const float spec = JoltProjectSettings::speculative_contact_distance;
	JPH::AABox box;
	for (int k = 0; k < p_soft_face.n; ++k) {
		box.Encapsulate(JPH::Vec3(p_base + p_soft_face.v[k]));
	}
	box.ExpandBy(JPH::Vec3::sReplicate(spec + 1e-3f));
	const JPH::Vec3 pf0 = p_partner_face.v[0];
	const float cos_1deg = 0.999847695f;
	JPH::Shape::GetTrianglesContext context;
	p_partner.GetShape()->GetTrianglesStart(context, box, p_partner.GetCenterOfMassPosition(), p_partner.GetRotation(), JPH::Vec3::sOne());
	constexpr int batch = JPH::Shape::cGetTrianglesMinTrianglesRequested;
	JPH::Float3 vertices[3 * batch];
	for (int guard = 0; guard < 64; ++guard) {
		const int count = p_partner.GetShape()->GetTrianglesNext(context, batch, vertices);
		if (count <= 0) {
			break;
		}
		for (int t = 0; t < count; ++t) {
			JPH::Vec3 v[3];
			double key[3] = { 0.0, 0.0, 0.0 };
			for (int c = 0; c < 3; ++c) {
				const JPH::Float3 &f = vertices[3 * t + c];
				key[0] += double(f.x) / 3.0;
				key[1] += double(f.y) / 3.0;
				key[2] += double(f.z) / 3.0;
				v[c] = JPH::Vec3(JPH::RVec3(JPH::Vec3(f)) - p_base);
			}
			JPH::Vec3 nt = (v[1] - v[0]).Cross(v[2] - v[0]);
			const float len = nt.Length();
			if (!(len > 0.0f)) {
				continue;
			}
			nt /= len;
			if (nt.Dot(p_out_normal) < cos_1deg || std::abs((v[0] - pf0).Dot(p_out_normal)) > spec) {
				continue;
			}
			Poly tri;
			for (int c = 0; c < 3; ++c) {
				tri.push((v[c] - p_o).Dot(p_t1), (v[c] - p_o).Dot(p_t2));
			}
			make_ccw(tri);
			Poly cut;
			clip_convex(p_soft_ccw, tri, cut);
			if (cut.n < 3 || !(std::abs(signed_area(cut)) > 1e-9) || r_set.n >= MAX_PARTNERS) {
				continue;
			}
			// insert in order of the world centroid (x, then y, then z)
			int at = r_set.n;
			while (at > 0) {
				const double *prev = r_set.key[at - 1];
				const bool after = prev[0] < key[0] || (prev[0] == key[0] && (prev[1] < key[1] || (prev[1] == key[1] && prev[2] <= key[2])));
				if (after) {
					break;
				}
				r_set.face[at] = r_set.face[at - 1];
				r_set.key[at][0] = prev[0];
				r_set.key[at][1] = prev[1];
				r_set.key[at][2] = prev[2];
				--at;
			}
			r_set.face[at] = tri;
			r_set.key[at][0] = key[0];
			r_set.key[at][1] = key[1];
			r_set.key[at][2] = key[2];
			++r_set.n;
		}
	}
}

void supporting_face(const JPH::Body &p_body, const JPH::SubShapeID &p_id, JPH::Vec3Arg p_outward_world, JPH::RVec3Arg p_base, SupportFace &r_face) {
	JPH::Shape::SupportingFace face;
	const JPH::Mat44 com = p_body.GetCenterOfMassTransform().PostTranslated(-p_base).ToMat44();
	// A supporting face is the one facing against the direction given, so its outward normal is minus that direction
	const JPH::Vec3 direction_local = p_body.GetRotation().Conjugated() * (-p_outward_world);
	p_body.GetShape()->GetSupportingFace(p_id, direction_local, JPH::Vec3::sOne(), com, face);
	r_face.n = MIN((int)face.size(), 32);
	for (int k = 0; k < r_face.n; ++k) {
		r_face.v[k] = face[k];
	}
}

// The height z(x, y) of a planar face along the normal, in the patch frame (origin o, axes t1, t2, n)
struct HeightPlane {
	double z0 = 0.0, gx = 0.0, gy = 0.0;
	bool valid = false;
};

HeightPlane face_plane(const SupportFace &p_face, const JPH::Vec3 &p_o, const JPH::Vec3 &p_t1, const JPH::Vec3 &p_t2, const JPH::Vec3 &p_n) {
	HeightPlane pl;
	if (p_face.n < 3) {
		return pl;
	}
	// Newell's normal and the centroid, in frame coordinates
	double nx = 0.0, ny = 0.0, nz = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
	double fx[32], fy[32], fz[32];
	for (int k = 0; k < p_face.n; ++k) {
		const JPH::Vec3 r = p_face.v[k] - p_o;
		fx[k] = r.Dot(p_t1);
		fy[k] = r.Dot(p_t2);
		fz[k] = r.Dot(p_n);
		cx += fx[k];
		cy += fy[k];
		cz += fz[k];
	}
	cx /= p_face.n;
	cy /= p_face.n;
	cz /= p_face.n;
	for (int k = 0; k < p_face.n; ++k) {
		const int k1 = (k + 1) % p_face.n;
		nx += (fy[k] - fy[k1]) * (fz[k] + fz[k1]);
		ny += (fz[k] - fz[k1]) * (fx[k] + fx[k1]);
		nz += (fx[k] - fx[k1]) * (fy[k] + fy[k1]);
	}
	const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
	if (!(len > 0.0) || std::abs(nz) < 0.2 * len) {
		return pl; // degenerate, or more than about 78 deg from the contact normal
	}
	pl.gx = -nx / nz;
	pl.gy = -ny / nz;
	pl.z0 = cz - pl.gx * cx - pl.gy * cy;
	pl.valid = true;
	return pl;
}

Poly project(const SupportFace &p_face, const JPH::Vec3 &p_o, const JPH::Vec3 &p_t1, const JPH::Vec3 &p_t2) {
	Poly poly;
	for (int k = 0; k < p_face.n; ++k) {
		const JPH::Vec3 r = p_face.v[k] - p_o;
		poly.push(r.Dot(p_t1), r.Dot(p_t2));
	}
	return poly;
}

// Brief 5.7: tangents from a body-2-fixed reference: body 2's local x projected on the patch plane, its local y when x
// is within 30 deg of the normal
void body_tangents(const JPH::Body &p_body2, const JPH::Vec3 &p_n, JPH::Vec3 &r_t1, JPH::Vec3 &r_t2) {
	const JPH::Quat rot = p_body2.GetRotation();
	JPH::Vec3 ref = rot * JPH::Vec3::sAxisX();
	if (std::abs(ref.Dot(p_n)) > 0.8660254037844386) {
		ref = rot * JPH::Vec3::sAxisY();
	}
	r_t1 = (ref - ref.Dot(p_n) * p_n).Normalized();
	r_t2 = p_n.Cross(r_t1);
}

// The kind, restitution, loss and damping of a pair: the lossier side's (CHOSEN in the reference)
const JoltSoftSurface3D &lossier(const JoltSoftSurface3D *p_s1, const JoltSoftSurface3D *p_s2) {
	if (p_s1 == nullptr) {
		return *p_s2;
	}
	if (p_s2 == nullptr) {
		return *p_s1;
	}
	return p_s1->restitution <= p_s2->restitution ? *p_s1 : *p_s2;
}

double rolling_arm(double p_m1, double p_loss, double p_additive) {
	return 0.5 * p_loss * p_m1 + p_additive;
}

// The first-touch law the solver reads (SoftContactPatch::mLawSquash, mLawForce): the patch's own pressed law sampled at
// p_n squashes log-spaced from p_lo to p_hi, read piecewise between them, so a first touch's predicted peak and a gap
// row's secant come from the law itself, not from one power law extrapolated across decades of squash. p_fn(s, F) gives
// the law's force at the squash s, or false where it cannot be read. With p_all, one sample that cannot be read leaves
// none (the caller falls back to another law); otherwise such a sample is left out. Returns the samples kept.
template <class Fn>
int set_law_samples(double p_lo, double p_hi, int p_n, bool p_all, Fn p_fn, JPH::SoftContactPatch &r_patch) {
	r_patch.mNumLawPoints = 0;
	const int n = MIN(p_n, (int)JPH::SoftContactPatch::cMaxLawPoints);
	if (!(p_lo > 0.0) || !(p_hi > p_lo) || n < 2) {
		return 0;
	}
	int kept = 0;
	for (int k = 0; k < n; ++k) {
		const double s = p_lo * std::pow(p_hi / p_lo, double(k) / double(n - 1));
		double F = 0.0;
		const bool ok = p_fn(s, F) && std::isfinite(F) && F > 0.0 && (kept == 0 || F > double(r_patch.mLawForce[kept - 1]));
		if (!ok) {
			if (p_all) {
				r_patch.mNumLawPoints = 0;
				return 0;
			}
			continue;
		}
		r_patch.mLawSquash[kept] = float(s);
		r_patch.mLawForce[kept] = float(F);
		++kept;
	}
	r_patch.mNumLawPoints = uint8_t(kept >= 2 ? kept : 0);
	return r_patch.mNumLawPoints;
}

// A linear law F = k s as the solver's two samples (exact at any squash)
void set_law_linear(double p_k, double p_scale, JPH::SoftContactPatch &r_patch) {
	r_patch.mNumLawPoints = 0;
	if (!(p_k > 0.0) || !(p_scale > 0.0)) {
		return;
	}
	r_patch.mLawSquash[0] = float(1e-3 * p_scale);
	r_patch.mLawForce[0] = float(p_k * 1e-3 * p_scale);
	r_patch.mLawSquash[1] = float(p_scale);
	r_patch.mLawForce[1] = float(p_k * p_scale);
	r_patch.mNumLawPoints = 2;
}

// The core guard's points of a face patch (brief 5.5, Stage 1c-b): every vertex of the overlap, at most the tail's room of
// them, spread evenly round it when there are more; each with the overlap of the undeformed surfaces there and the layer
// left before the cores meet
template <class World>
void set_face_cores(const Poly &p_overlap, double p_d0, double p_gx, double p_gy, double p_layer, World p_world, JPH::SoftContactPatch &r_patch) {
	const int room = JPH::SoftContactTail::cMaxCores;
	const int n = p_overlap.n;
	const int count = MIN(n, room);
	r_patch.mNumCores = 0;
	for (int i = 0; i < count; ++i) {
		const int k = n <= room ? i : (i * n) / room;
		const double x = p_overlap.x[k], y = p_overlap.y[k];
		const double d = p_d0 + p_gx * x + p_gy * y;
		r_patch.mCorePoint[i] = p_world(x, y);
		r_patch.mCoreSquash[i] = float(d);
		r_patch.mCoreGapAt[i] = float(p_layer - d);
	}
	r_patch.mNumCores = uint8_t(count);
}

} // namespace

bool JoltSoftSurface3D::from_params(const PackedFloat32Array &p_params, JoltSoftSurface3D &r_surface, String &r_error) {
	r_surface = JoltSoftSurface3D();
	if (p_params.is_empty()) {
		return true;
	}
	if (p_params.size() < 12) {
		r_error = vformat("the parameter array has %d values, at least 12 are needed (kind, modulus, bulk modulus, thickness, lumped stiffness, travel, three damping ratios, loss, restitution, sliding friction)", p_params.size());
		return false;
	}
	const float *p = p_params.ptr();
	for (int i = 0; i < p_params.size(); ++i) {
		if (!std::isfinite(p[i])) {
			r_error = vformat("value %d is not finite", i);
			return false;
		}
	}
	r_surface.kind = int(p[0]);
	r_surface.modulus = p[1];
	r_surface.bulk_modulus = p[2];
	r_surface.thickness = p[3];
	r_surface.lumped_stiffness = p[4];
	r_surface.travel = p[5];
	r_surface.zeta[0] = p[6];
	r_surface.zeta[1] = p[7];
	r_surface.zeta[2] = p[8];
	r_surface.loss = p[9];
	r_surface.restitution = p[10];
	r_surface.sliding_friction = p[11];
	r_surface.rolling_arm = p_params.size() > 12 ? MAX(double(p[12]), 0.0) : 0.0;
	if (r_surface.kind == KIND_NONE) {
		return true;
	}
	if (r_surface.kind < KIND_LAYER || r_surface.kind > KIND_LUMPED) {
		r_error = vformat("the kind %d is none of 0 (none), 1 (layer), 2 (solid), 3 (lumped)", r_surface.kind);
		return false;
	}
	if (r_surface.restitution < 0.0 || r_surface.restitution > 1.0 || r_surface.loss < 0.0 || r_surface.loss > 1.0 || r_surface.zeta[0] < 0.0 || r_surface.zeta[1] < 0.0 || r_surface.zeta[2] < 0.0) {
		r_error = "the restitution and the loss must lie in [0, 1] and the damping ratios must not be negative";
		return false;
	}
	if (r_surface.kind == KIND_LUMPED) {
		if (!(r_surface.lumped_stiffness > 0.0)) {
			r_error = "a lumped surface needs a positive lumped stiffness";
			return false;
		}
		return true;
	}
	if (!(r_surface.modulus > 0.0) || !(9.0 * r_surface.bulk_modulus > r_surface.modulus)) {
		r_error = "the modulus must be positive and the bulk modulus above a ninth of it (a positive shear modulus)";
		return false;
	}
	if (r_surface.kind == KIND_LAYER && !(r_surface.thickness > 0.0)) {
		r_error = "a layer needs a positive thickness";
		return false;
	}
	return true;
}

double JoltSoftSurface3D::zeta_for(double p_exponent) const {
	const double n = MIN(MAX(p_exponent, 1.0), 2.0);
	if (n <= 1.5) {
		const double t = (n - 1.0) / 0.5;
		return (1.0 - t) * zeta[0] + t * zeta[1];
	}
	const double t = (n - 1.5) / 0.5;
	return (1.0 - t) * zeta[1] + t * zeta[2];
}

bool jolt_build_soft_patch(const JoltSoftPatchInput &p_in, JPH::SoftContactPatch &r_patch, JoltSoftBuildResult &r_result) {
	r_patch = JPH::SoftContactPatch();
	r_result = JoltSoftBuildResult();
	const JPH::Body &body1 = *p_in.body1;
	const JPH::Body &body2 = *p_in.body2;
	const JPH::ContactManifold &manifold = *p_in.manifold;
	const JoltSoftSurface3D *s1 = p_in.surface1 != nullptr && p_in.surface1->kind != JoltSoftSurface3D::KIND_NONE ? p_in.surface1 : nullptr;
	const JoltSoftSurface3D *s2 = p_in.surface2 != nullptr && p_in.surface2->kind != JoltSoftSurface3D::KIND_NONE ? p_in.surface2 : nullptr;
	if (s1 == nullptr && s2 == nullptr) {
		return false;
	}
	// The soft side whose law leads (the first soft one), and the partner's surface (nullptr when rigid)
	const bool soft_is_1 = s1 != nullptr;
	const JoltSoftSurface3D &soft = soft_is_1 ? *s1 : *s2;
	const JoltSoftSurface3D *partner = soft_is_1 ? s2 : s1;
	const bool two_layers = partner != nullptr && soft.kind == JoltSoftSurface3D::KIND_LAYER && partner->kind == JoltSoftSurface3D::KIND_LAYER;

	const JPH::Vec3 n = manifold.mWorldSpaceNormal;
	const JPH::RVec3 base = manifold.mBaseOffset;

	// The two supporting faces along the normal: body 1's faces +n, body 2's faces -n
	SupportFace face1, face2;
	supporting_face(body1, manifold.mSubShapeID1, n, base, face1);
	supporting_face(body2, manifold.mSubShapeID2, -n, base, face2);

	// The frame's origin: the mean of the manifold's points, midway between the surfaces
	JPH::Vec3 o = JPH::Vec3::sZero();
	const int npts = (int)manifold.mRelativeContactPointsOn1.size();
	for (int k = 0; k < npts; ++k) {
		o += 0.5f * (manifold.mRelativeContactPointsOn1[k] + manifold.mRelativeContactPointsOn2[k]);
	}
	if (npts > 0) {
		o /= float(npts);
	}

	const JoltSoftSurface3D &loss_side = lossier(s1, s2);
	r_patch.mRestitution = float(loss_side.restitution);
	r_patch.mDampingRatio = float(loss_side.zeta[0]);
	for (int i = 0; i < 3; ++i) {
		r_patch.mLawDampingRatio[i] = float(loss_side.zeta[i]);
	}
	// The first-touch law is read only for a first touch still to be decided and for a patch across a gap (Stage 1c-b)
	const bool onset_pending = p_in.tail == nullptr || (p_in.tail->mFlags & JPH::SoftContactTail::OnsetPending) != 0;
	r_patch.mStaticFriction = p_in.static_friction;
	r_patch.mSlidingFriction = MIN(p_in.sliding_friction, p_in.static_friction);
	const double additive_arm = (s1 != nullptr ? s1->rolling_arm : 0.0) + (s2 != nullptr ? s2->rolling_arm : 0.0);
	const double mu = p_in.static_friction;

	// Class by the faces' vertex counts: Face (both 3 or more), Line (one side 2, the other a face), Point (one side
	// 0 or 1, the other a face)
	int cls = -1;
	const SupportFace *flat = nullptr;
	const SupportFace *curved = nullptr;
	bool curved_is_1 = false;
	if (face1.n >= 3 && face2.n >= 3) {
		cls = 0;
	} else if (face1.n >= 3 && face2.n <= 2) {
		flat = &face1;
		curved = &face2;
		curved_is_1 = false;
		cls = face2.n == 2 ? 1 : 2;
	} else if (face2.n >= 3 && face1.n <= 2) {
		flat = &face2;
		curved = &face1;
		curved_is_1 = true;
		cls = face1.n == 2 ? 1 : 2;
	}
	if (cls < 0) {
		r_result.outcome = JoltSoftBuildResult::UNCLASSED;
		return false;
	}
	r_patch.mClass = uint8_t(cls);

	if (cls == 0) {
		// ---------------------------------------------------------------- Face
		if (soft.kind == JoltSoftSurface3D::KIND_SOLID) {
			r_result.outcome = JoltSoftBuildResult::SOLID_FACE;
			return false;
		}
		JPH::Vec3 t1, t2;
		body_tangents(body2, n, t1, t2);
		const HeightPlane pl1 = face_plane(face1, o, t1, t2, n);
		const HeightPlane pl2 = face_plane(face2, o, t1, t2, n);
		if (!pl1.valid || !pl2.valid) {
			r_result.outcome = JoltSoftBuildResult::UNCLASSED;
			return false;
		}
		// The overlap of the undeformed surfaces: body 1's surface height less body 2's
		const double d0 = pl1.z0 - pl2.z0;
		const double gx = pl1.gx - pl2.gx;
		const double gy = pl1.gy - pl2.gy;
		const Poly poly1 = project(face1, o, t1, t2);
		const Poly poly2 = project(face2, o, t1, t2);
		const Poly &soft_poly = soft_is_1 ? poly1 : poly2;
		const Poly &partner_poly = soft_is_1 ? poly2 : poly1;

		// The deepest point of the footprints' overlap (the average of its deepest vertices)
		Poly overlap;
		clip_convex(soft_poly, partner_poly, overlap);
		if (overlap.n < 3) {
			r_result.outcome = JoltSoftBuildResult::NO_OVERLAP;
			return false;
		}
		double dmax = -1e300;
		for (int k = 0; k < overlap.n; ++k) {
			dmax = MAX(dmax, d0 + gx * overlap.x[k] + gy * overlap.y[k]);
		}
		double ddx = 0.0, ddy = 0.0, dcount = 0.0;
		for (int k = 0; k < overlap.n; ++k) {
			if (d0 + gx * overlap.x[k] + gy * overlap.y[k] >= dmax - 1e-6) {
				ddx += overlap.x[k];
				ddy += overlap.y[k];
				dcount += 1.0;
			}
		}
		ddx /= dcount;
		ddy /= dcount;
		auto world = [&](double p_x, double p_y) {
			const double z1 = pl1.z0 + pl1.gx * p_x + pl1.gy * p_y;
			const double z2 = pl2.z0 + pl2.gx * p_x + pl2.gy * p_y;
			return base + o + float(p_x) * t1 + float(p_y) * t2 + float(0.5 * (z1 + z2)) * n;
		};
		r_patch.mDeepest = world(ddx, ddy);
		r_patch.mSquash = float(dmax);
		r_patch.mTangent1 = t1;
		r_patch.mTangent2 = t2;

		// Brief 5.2: against a partner of several coplanar faces (a mesh's triangles, a compound's children) the footprint
		// is one field, not one patch per face: each would take the rest of the layer as free and read it soft. The
		// manifold on the leading face (the one under the soft face's centroid; the first in the fixed order when several
		// hold it, or when none does) builds the field over every coplanar face under the footprint; the others keep only
		// their core guard. A partner face that already holds every coplanar face (a box's quad over its two
		// triangles) is the ordinary case.
		PartnerSet partners;
		partners.n = 1;
		partners.face[0] = partner_poly;
		make_ccw(partners.face[0]);
		bool follower = false;
		if (soft.kind == JoltSoftSurface3D::KIND_LAYER || soft.kind == JoltSoftSurface3D::KIND_LUMPED) {
			thread_local PartnerSet scratch_set;
			PartnerSet *set = &scratch_set;
			Poly soft_ccw = soft_poly;
			make_ccw(soft_ccw);
			coplanar_faces(soft_is_1 ? body2 : body1, soft_is_1 ? -n : n, soft_is_1 ? face2 : face1, soft_is_1 ? face1 : face2, soft_ccw, base, o, t1, t2, *set);
			bool covers = true;
			for (int i = 0; i < set->n && covers; ++i) {
				double mx = 0.0, my = 0.0;
				for (int k = 0; k < set->face[i].n; ++k) {
					mx += set->face[i].x[k];
					my += set->face[i].y[k];
				}
				covers = inside_convex_tol(partners.face[0], mx / set->face[i].n, my / set->face[i].n, 1e-5);
			}
			if (set->n >= 2 && !covers) {
				const PolyMoments sm = poly_moments(soft_ccw);
				const double scx = sm.A > 0.0 ? sm.Sx / sm.A : 0.0;
				const double scy = sm.A > 0.0 ? sm.Sy / sm.A : 0.0;
				int leader = 0;
				for (int i = 0; i < set->n; ++i) {
					if (inside_convex_tol(set->face[i], scx, scy, 1e-5)) {
						leader = i;
						break;
					}
				}
				double lx = 0.0, ly = 0.0;
				for (int k = 0; k < set->face[leader].n; ++k) {
					lx += set->face[leader].x[k];
					ly += set->face[leader].y[k];
				}
				if (inside_convex_tol(partners.face[0], lx / set->face[leader].n, ly / set->face[leader].n, 1e-5)) {
					partners = *set;
				} else {
					follower = true;
				}
			}
		}
		if (follower) {
			const double follower_layer = soft.kind == JoltSoftSurface3D::KIND_LAYER ? (two_layers ? s1->thickness + s2->thickness : soft.thickness) : (soft.travel > 0.0 ? soft.travel : 1e6);
			r_patch.mForce = 0.0f;
			r_patch.mNormalStiffness = 0.0f;
			r_patch.mLawCoefficient = 0.0f;
			r_patch.mLawExponent = 1.0f;
			r_patch.mStaticFriction = 0.0f;
			r_patch.mSlidingFriction = 0.0f;
			r_patch.mCentre = r_patch.mDeepest;
			r_patch.mCoreGap = float(follower_layer - dmax);
			set_face_cores(overlap, d0, gx, gy, follower_layer, world, r_patch);
			r_patch.mNumTilt = 0;
			r_patch.mNumRoll = 0;
			r_patch.mRigidStick = true;
			r_patch.mEnabled = true;
			r_result.outcome = JoltSoftBuildResult::BUILT;
			return true;
		}
		if (partners.n > 1) {
			// The leading face's deepest point is the whole footprint's (a level contact: the footprint's centroid), so a
			// first touch's impact row and the core guard act where the whole field does, not off at its own face
			thread_local Poly scratch_pieces[MAX_PARTNERS];
			Poly *pieces = scratch_pieces;
			int num_pieces = 0;
			double dm = -1e300;
			for (int i = 0; i < partners.n; ++i) {
				clip_convex(soft_poly, partners.face[i], pieces[num_pieces]);
				if (pieces[num_pieces].n < 3) {
					continue;
				}
				for (int k = 0; k < pieces[num_pieces].n; ++k) {
					dm = MAX(dm, d0 + gx * pieces[num_pieces].x[k] + gy * pieces[num_pieces].y[k]);
				}
				++num_pieces;
			}
			if (num_pieces > 0) {
				bool level = true;
				double sx = 0.0, sy = 0.0, cnt = 0.0, ua = 0.0, usx = 0.0, usy = 0.0;
				for (int i = 0; i < num_pieces; ++i) {
					make_ccw(pieces[i]);
					const PolyMoments pm = poly_moments(pieces[i]);
					ua += pm.A;
					usx += pm.Sx;
					usy += pm.Sy;
					for (int k = 0; k < pieces[i].n; ++k) {
						if (d0 + gx * pieces[i].x[k] + gy * pieces[i].y[k] >= dm - 1e-6) {
							sx += pieces[i].x[k];
							sy += pieces[i].y[k];
							cnt += 1.0;
						} else {
							level = false;
						}
					}
				}
				if (level && ua > 0.0) {
					ddx = usx / ua;
					ddy = usy / ua;
				} else if (cnt > 0.0) {
					ddx = sx / cnt;
					ddy = sy / cnt;
				}
				dmax = dm;
				r_patch.mDeepest = world(ddx, ddy);
				r_patch.mSquash = float(dmax);
			}
		}

		double layer = 0.0;
		Coeffs co;
		double share = 1.0; // two layers: the softer side's share of the squash
		bool lubricated_series = false;
		if (soft.kind == JoltSoftSurface3D::KIND_LAYER) {
			if (two_layers) {
				// The lead's decision 5: each side's layer with a lubricated face, compliances in series
				const Coeffs c1 = split_coeffs(s1->modulus, s1->bulk_modulus, s1->thickness, true);
				const Coeffs c2 = split_coeffs(s2->modulus, s2->bulk_modulus, s2->thickness, true);
				thread_local FaceLaw scratch_l1;
				thread_local FaceLaw scratch_l2;
				FaceLaw *l1 = &scratch_l1;
				FaceLaw *l2 = &scratch_l2;
				const bool ok1 = face_law(soft_poly, partners, 1.0, 0.0, 0.0, c1, nullptr, *l1);
				const bool ok2 = face_law(soft_poly, partners, 1.0, 0.0, 0.0, c2, nullptr, *l2);
				if (ok1 && ok2 && l1->k_n > 0.0 && l2->k_n > 0.0) {
					const double k1 = l1->k_n, k2 = l2->k_n;
					share = (1.0 / (1.0 / k1 + 1.0 / k2)) / MIN(k1, k2);
					co = k1 <= k2 ? c1 : c2;
				} else {
					co = c1;
				}
				lubricated_series = true;
				layer = s1->thickness + s2->thickness;
			} else {
				co = split_coeffs(soft.modulus, soft.bulk_modulus, soft.thickness, false);
				layer = soft.thickness;
			}
		}

		thread_local FaceLaw scratch_law;
		FaceLaw *law = &scratch_law;
		bool pressed = false;
		if (soft.kind == JoltSoftSurface3D::KIND_LAYER) {
			pressed = face_law(soft_poly, partners, d0 * share, gx * share, gy * share, co, p_in.tail, *law) && !law->empty;
		}
		double F = 0.0, kn = 0.0, cx = ddx, cy = ddy;
		double ratio = 1.0;
		if (soft.kind == JoltSoftSurface3D::KIND_LUMPED) {
			// A lumped body against a face: the flat's bed-of-springs profile (design 2.4), springs of K_l / A per area
			// over the overlap A, so a level squash d carries F = K_l d. The pressure follows the local squash: the centre
			// of pressure moves with tilt, and the face resists tilt with its tilt rows (k_a times the active part's
			// second moments about the centre), as a layer face does.
			// The overlap with each partner face (one, or the coplanar ones of brief 5.2), and its pressed part
			double whole_A = 0.0;
			PolyMoments m;
			thread_local Poly scratch_acts[MAX_PARTNERS];
			Poly *acts = scratch_acts;
			int num_acts = 0;
			for (int i = 0; i < partners.n; ++i) {
				Poly ov;
				clip_convex(soft_poly, partners.face[i], ov);
				if (ov.n < 3) {
					continue;
				}
				make_ccw(ov);
				whole_A += poly_moments(ov).A;
				Poly act;
				clip_halfplane(ov, gx, gy, d0, act);
				if (act.n >= 3) {
					const PolyMoments mi = poly_moments(act);
					m.A += mi.A;
					m.Sx += mi.Sx;
					m.Sy += mi.Sy;
					m.Sxx += mi.Sxx;
					m.Syy += mi.Syy;
					m.Sxy += mi.Sxy;
					acts[num_acts++] = act;
				}
			}
			const double ka = whole_A > 0.0 ? soft.lumped_stiffness / whole_A : 0.0;
			kn = soft.lumped_stiffness;
			layer = soft.travel > 0.0 ? soft.travel : 1e6;
			r_patch.mArea = float(whole_A);
			r_patch.mSpinRadius = float(std::sqrt(whole_A / PI) * 2.0 / 3.0);
			if (num_acts > 0 && ka > 0.0) {
				const double W = d0 * m.A + gx * m.Sx + gy * m.Sy;
				if (m.A > 0.0 && W > 0.0) {
					cx = (d0 * m.Sx + gx * m.Sxx + gy * m.Sxy) / W;
					cy = (d0 * m.Sy + gx * m.Sxy + gy * m.Syy) / W;
					F = ka * W;
					kn = ka * m.A;
					const double kt_x = ka * (m.Syy - 2.0 * cy * m.Sy + cy * cy * m.A);
					const double kt_y = ka * (m.Sxx - 2.0 * cx * m.Sx + cx * cx * m.A);
					const double kt_xy = ka * (m.Sxy - cx * m.Sy - cy * m.Sx + cx * cy * m.A);
					// The spin radius and the first moments of the pressure, by quadrature over a fan of each active piece
					double rs = 0.0, m1x = 0.0, m1y = 0.0;
					for (int j = 0; j < num_acts; ++j) {
						const Poly &act = acts[j];
						for (int t = 1; t + 1 < act.n; ++t) {
							const double x0 = act.x[0], y0 = act.y[0], x1 = act.x[t], y1 = act.y[t], x2 = act.x[t + 1], y2 = act.y[t + 1];
							const double at = 0.5 * std::abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
							for (int q = 0; q < 7; ++q) {
								const double c = 1.0 - TRI_QA[q] - TRI_QB[q];
								const double px = TRI_QA[q] * x0 + TRI_QB[q] * x1 + c * x2;
								const double py = TRI_QA[q] * y0 + TRI_QB[q] * y1 + c * y2;
								const double wq = at * TRI_QW[q] * MAX(d0 + gx * px + gy * py, 0.0);
								rs += wq * std::sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
								m1x += wq * std::abs(px - cx);
								m1y += wq * std::abs(py - cy);
							}
						}
					}
					r_patch.mSpinRadius = float(rs / W);
					r_patch.mArea = float(m.A);
					r_result.m1[0] = m1x / W;
					r_result.m1[1] = m1y / W;
					// Tilt axes as a layer face's (brief 5.7), the arms to the active part's edge on each side
					const double tr = 0.5 * (kt_x + kt_y);
					const double df = 0.5 * (kt_x - kt_y);
					const double rad = std::sqrt(df * df + kt_xy * kt_xy);
					const double ev_lo = tr - rad, ev_hi = tr + rad;
					double ax_x[2], ax_y[2], stiff[2];
					if (!(ev_hi > 0.0) || (ev_hi - ev_lo) / ev_hi <= 0.10) {
						ax_x[0] = 1.0;
						ax_y[0] = 0.0;
						ax_x[1] = 0.0;
						ax_y[1] = 1.0;
						stiff[0] = kt_x;
						stiff[1] = kt_y;
					} else {
						double vx, vy;
						if (std::abs(kt_xy) > 0.0) {
							vx = kt_xy;
							vy = ev_lo - kt_x;
						} else if (kt_x <= kt_y) {
							vx = 1.0;
							vy = 0.0;
						} else {
							vx = 0.0;
							vy = 1.0;
						}
						const double vl = std::sqrt(vx * vx + vy * vy);
						ax_x[0] = vx / vl;
						ax_y[0] = vy / vl;
						ax_x[1] = -ax_y[0];
						ax_y[1] = ax_x[0];
						stiff[0] = ev_lo;
						stiff[1] = ev_hi;
					}
					r_patch.mNumTilt = 2;
					for (int i = 0; i < 2; ++i) {
						r_patch.mTiltAxis[i] = float(ax_x[i]) * t1 + float(ax_y[i]) * t2;
						r_patch.mTiltStiffness[i] = float(MAX(stiff[i], 0.0));
						const double mx = -ax_y[i], my = ax_x[i];
						double pos = 0.0, neg = 0.0;
						for (int j = 0; j < num_acts; ++j) {
							for (int k = 0; k < acts[j].n; ++k) {
								const double s = (acts[j].x[k] - cx) * mx + (acts[j].y[k] - cy) * my;
								pos = MAX(pos, s);
								neg = MAX(neg, -s);
							}
						}
						r_patch.mTiltArmPos[i] = float(pos);
						r_patch.mTiltArmNeg[i] = float(neg);
					}
				}
			}
			if (!(F > 0.0)) {
				cx = ddx;
				cy = ddy;
			}
			r_patch.mLawCoefficient = float(soft.lumped_stiffness);
			r_patch.mLawExponent = 1.0f;
			if (onset_pending || !(F > 0.0)) {
				set_law_linear(soft.lumped_stiffness, MIN(layer, 1.0), r_patch);
			}
			r_patch.mNormalDampingRatio = float(loss_side.zeta_for(1.0));
			r_patch.mRigidStick = true;
		} else if (pressed) {
			r_result.passes = law->full.passes;
			r_result.not_settled = !law->full.settled;
			r_result.raw_force = law->F;
			double kt_x = law->kt_x, kt_y = law->kt_y, kt_xy = law->kt_xy;
			kn = law->k_n;
			F = law->F;
			if (lubricated_series) {
				kn *= share;
				kt_x *= share;
				kt_y *= share;
				kt_xy *= share;
			} else {
				// Brief 3.2: the field scaled by T3 at the patch's shape factor and the pair's static mu
				ratio = table_ratio(false, shape_factor(law->area, law->Ixx, law->Iyy, soft.thickness), mu, kg_of(soft));
				F *= ratio;
				kn *= ratio;
				kt_x *= ratio;
				kt_y *= ratio;
				kt_xy *= ratio;
			}
			cx = law->cx;
			cy = law->cy;
			r_patch.mSpinRadius = float(law->r_s);
			r_patch.mArea = float(law->area);
			r_result.m1[0] = law->m1x;
			r_result.m1[1] = law->m1y;

			// Tilt axes (brief 5.7): the field's principal axes only when the two tilt stiffnesses differ by more than
			// 10 %, else the body-fixed tangents. Stiffness about an axis a, moment arms along n x a.
			const double tr = 0.5 * (kt_x + kt_y);
			const double df = 0.5 * (kt_x - kt_y);
			const double rad = std::sqrt(df * df + kt_xy * kt_xy);
			const double ev_lo = tr - rad, ev_hi = tr + rad;
			double ax_x[2], ax_y[2], stiff[2];
			if (!(ev_hi > 0.0) || (ev_hi - ev_lo) / ev_hi <= 0.10) {
				ax_x[0] = 1.0;
				ax_y[0] = 0.0;
				ax_x[1] = 0.0;
				ax_y[1] = 1.0;
				stiff[0] = kt_x;
				stiff[1] = kt_y;
			} else {
				// the eigenvector of the smaller eigenvalue first, the other n x it
				double vx, vy;
				if (std::abs(kt_xy) > 0.0) {
					vx = kt_xy;
					vy = ev_lo - kt_x;
				} else if (kt_x <= kt_y) {
					vx = 1.0;
					vy = 0.0;
				} else {
					vx = 0.0;
					vy = 1.0;
				}
				const double vl = std::sqrt(vx * vx + vy * vy);
				ax_x[0] = vx / vl;
				ax_y[0] = vy / vl;
				ax_x[1] = -ax_y[0];
				ax_y[1] = ax_x[0];
				stiff[0] = ev_lo;
				stiff[1] = ev_hi;
			}
			r_patch.mNumTilt = 2;
			for (int i = 0; i < 2; ++i) {
				r_patch.mTiltAxis[i] = float(ax_x[i]) * t1 + float(ax_y[i]) * t2;
				r_patch.mTiltStiffness[i] = float(MAX(stiff[i], 0.0));
				// a positive moment about a shifts the pressure along n x a = (-a_y, a_x) in the frame
				const double mx = -ax_y[i], my = ax_x[i];
				r_patch.mTiltArmPos[i] = float(edge_distance(law->full, mx, my));
				r_patch.mTiltArmNeg[i] = float(edge_distance(law->full, -mx, -my));
			}

			// The tail's contact set for the next step's first pass
			const Grid &g = law->full.grid;
			for (int k = 0; k < g.n() && k < 192; ++k) {
				if (law->full.active[k]) {
					r_patch.mActiveSet[k >> 5] |= 1u << (k & 31);
				}
			}
			r_patch.mGridLong = uint8_t(g.nl);
			r_patch.mGridShort = uint8_t(g.ns);

			// Pre-slip shear (Layer): G A / h per tangent, G J_p / h about the normal
			r_patch.mShearStiffness = float(co.G * law->area / soft.thickness);
			r_patch.mSpinStiffness = float(co.G * law->J_p / soft.thickness);
			r_patch.mRelaxationLength = float(0.5 * (law->full.edge_x[1] - law->full.edge_x[0]));
			r_patch.mNormalDampingRatio = float(loss_side.zeta_for(1.0));
		}

		if (soft.kind == JoltSoftSurface3D::KIND_LAYER) {
			// The law for a first touch and across a gap: the normal stiffness of the footprints in full contact. Only a
			// gap or an undecided first touch needs it; a pressed patch whose first touch is decided keeps its own.
			double k_ref = kn;
			if (!pressed || onset_pending) {
				thread_local FaceLaw scratch_ref;
				FaceLaw *ref = &scratch_ref;
				if (face_law(soft_poly, partners, 1.0 * share, 0.0, 0.0, co, nullptr, *ref) && !ref->empty) {
					k_ref = ref->k_n * (lubricated_series ? share : table_ratio(false, shape_factor(ref->area, ref->Ixx, ref->Iyy, soft.thickness), mu, kg_of(soft)));
				}
				set_law_linear(k_ref, layer, r_patch);
			}
			r_patch.mLawCoefficient = float(k_ref);
			r_patch.mLawExponent = 1.0f;
			if (!pressed) {
				kn = k_ref;
				r_patch.mNormalDampingRatio = float(loss_side.zeta_for(1.0));
				r_patch.mShearStiffness = float(co.G * std::abs(signed_area(overlap)) / soft.thickness);
				r_patch.mSpinStiffness = 0.0f;
				r_patch.mNumTilt = 0;
			}
		}

		r_result.face_ratio = ratio;
		r_patch.mForce = float(MAX(F, 0.0));
		r_patch.mNormalStiffness = float(MAX(kn, 0.0));
		r_patch.mCentre = F > 0.0 ? world(cx, cy) : r_patch.mDeepest;
		r_patch.mCoreGap = float(layer - dmax);
		set_face_cores(overlap, d0, gx, gy, layer, world, r_patch);
		r_patch.mNumRoll = 0;
		r_patch.mRollingRadius = 0.0f;
		r_patch.mEnabled = true;
		r_result.outcome = JoltSoftBuildResult::BUILT;
		return true;
	}

	// ---------------------------------------------------------------- Line and Point
	const JPH::Body &curved_body = curved_is_1 ? body1 : body2;
	const JPH::SubShapeID curved_id = curved_is_1 ? manifold.mSubShapeID1 : manifold.mSubShapeID2;
	JPH::Vec3 leaf_scale;
	const JPH::Shape *leaf = leaf_shape(curved_body.GetShape(), curved_id, leaf_scale);
	double R = leaf_radius(leaf, leaf_scale);
	if (cls == 1 && !(R > 0.0)) {
		// a polyhedral edge: rounded to the layer's depth (design 4.1 item 2)
		R = soft.kind == JoltSoftSurface3D::KIND_LAYER ? soft.thickness : 0.0;
	}
	if (!(R > 0.0)) {
		r_result.outcome = JoltSoftBuildResult::UNCLASSED;
		return false;
	}

	JPH::Vec3 t1, t2;
	double s0 = 0.0, camber = 0.0, ya = 0.0, yb = 0.0;
	JPH::Vec3 origin = o;
	HeightPlane flat_plane;
	if (cls == 1) {
		// t2 along the line, t1 across it (the rolling direction): t2 = n x t1
		JPH::Vec3 along = curved->v[1] - curved->v[0];
		along = along - along.Dot(n) * n;
		if (!(along.Length() > 0.0f)) {
			r_result.outcome = JoltSoftBuildResult::UNCLASSED;
			return false;
		}
		t2 = along.Normalized();
		t1 = t2.Cross(n);
		origin = 0.5f * (curved->v[0] + curved->v[1]);
		flat_plane = face_plane(*flat, origin, t1, t2, n);
		if (!flat_plane.valid) {
			r_result.outcome = JoltSoftBuildResult::UNCLASSED;
			return false;
		}
		// The line's ends and its height along it
		const double y0 = (curved->v[0] - origin).Dot(t2), y1 = (curved->v[1] - origin).Dot(t2);
		const double z0 = (curved->v[0] - origin).Dot(n), z1 = (curved->v[1] - origin).Dot(n);
		const double line_slope = (z1 - z0) / (y1 - y0);
		const double line_z0 = z0 - line_slope * y0;
		// The overlap d = (body 1's height) - (body 2's height) along the line (x = 0)
		const double z_line0 = line_z0, z_face0 = flat_plane.z0;
		const double sz_line = line_slope, sz_face = flat_plane.gy;
		if (curved_is_1) {
			s0 = z_line0 - z_face0;
			camber = sz_line - sz_face;
		} else {
			s0 = z_face0 - z_line0;
			camber = sz_face - sz_line;
		}
		// The loaded length: the line's extent within the flat face's footprint along x = 0
		Poly flat_poly = project(*flat, origin, t1, t2);
		make_ccw(flat_poly);
		Poly strip = rect(-1e-6, 1e-6, MIN(y0, y1), MAX(y0, y1));
		Poly cut;
		clip_convex(strip, flat_poly, cut);
		if (cut.n < 3) {
			r_result.outcome = JoltSoftBuildResult::NO_OVERLAP;
			return false;
		}
		double cx0, cx1;
		poly_bounds(cut, cx0, cx1, ya, yb);
		if (!(yb > ya)) {
			r_result.outcome = JoltSoftBuildResult::NO_OVERLAP;
			return false;
		}
	} else {
		body_tangents(body2, n, t1, t2);
		// A point: the overlap is the manifold's penetration, at the midpoint of its deepest point pair
		s0 = manifold.mPenetrationDepth;
		origin = o;
	}

	// The deepest point and the patch frame's axes
	const double y_deep = cls == 1 ? (camber >= 0.0 ? yb : ya) : 0.0;
	const double d_deep = cls == 1 ? s0 + camber * y_deep : s0;
	r_patch.mTangent1 = t1;
	r_patch.mTangent2 = t2;
	r_patch.mSquash = float(d_deep);
	auto line_world = [&](double p_x, double p_y) {
		if (cls == 1) {
			const double zf = flat_plane.z0 + flat_plane.gx * p_x + flat_plane.gy * p_y;
			const double zl = zf + (curved_is_1 ? 1.0 : -1.0) * (s0 + camber * p_y);
			return base + origin + float(p_x) * t1 + float(p_y) * t2 + float(0.5 * (zf + zl)) * n;
		}
		return base + origin + float(p_x) * t1 + float(p_y) * t2;
	};
	r_patch.mDeepest = line_world(0.0, y_deep);

	// A Line spreads along it: the centroid of the overlap it would have one speculative band (w) deeper than now,
	// d(y) - d_deep + w, a level line's middle and a cambered one's toward the end that touches first. A first touch solved
	// as an impact acts there (at the deeper end itself it would turn a level line about its far end), and so does the row
	// of a Line that presses nothing yet. A Point's is its deepest point.
	double y_band = y_deep;
	if (cls == 1) {
		const double band = MAX(double(JoltProjectSettings::speculative_contact_distance), 1e-9);
		const double dy = (yb - ya) / N_SLICES;
		double sp = 0.0, spy = 0.0;
		for (int i = 0; i < N_SLICES; ++i) {
			const double y = ya + (double(i) + 0.5) * dy;
			const double p = MAX(camber * (y - y_deep) + band, 0.0);
			sp += p;
			spy += p * y;
		}
		if (sp > 0.0) {
			y_band = spy / sp;
		}
		r_patch.mHasImpactPoint = true;
		r_patch.mImpactPoint = line_world(0.0, y_band);
		r_patch.mImpactSquash = float(s0 + camber * y_band);
	}

	CurvedFig fig;
	bool have = false;
	double layer = 0.0;
	bool rigid_stick = true;
	double shear_G = 0.0;
	if (soft.kind == JoltSoftSurface3D::KIND_LAYER) {
		layer = two_layers ? s1->thickness + s2->thickness : soft.thickness;
		rigid_stick = false;
		const Coeffs co = two_layers ? two_sided_design(*s1, *s2) : split_coeffs(soft.modulus, soft.bulk_modulus, soft.thickness, false);
		shear_G = co.G;
		if (s0 > 0.0 || (cls == 1 && d_deep > 0.0)) {
			if (!two_layers) {
				have = cls == 1 ? table_line(s0, camber, ya, yb, R, soft, mu, fig) : table_point(s0, R, soft, mu, fig);
			}
			if (!have) {
				have = curved_outside(cls == 1, s0, R, co, camber, ya, yb, fig);
				if (have && !fig.empty) {
					fig.face_ratio = table_ratio(false, shape_factor(fig.area, fig.Ixx, fig.Iyy, soft.thickness), mu, kg_of(soft));
					fig.F *= fig.face_ratio;
					fig.k_n *= fig.face_ratio;
					fig.kt_x *= fig.face_ratio;
					r_result.outside_table = true;
				}
			}
		}
		// The law from first touch: the table at a reference squash of a thousandth of the depth
		CurvedFig ref;
		const double s_ref = 1e-3 * soft.thickness;
		bool ref_ok = !two_layers && (cls == 1 ? table_line(s_ref, 0.0, ya, yb, R, soft, mu, ref) : table_point(s_ref, R, soft, mu, ref));
		if (ref_ok && !ref.empty && ref.F > 0.0) {
			const double nexp = MAX(ref.n_exp, 1.0);
			r_patch.mLawExponent = float(nexp);
			r_patch.mLawCoefficient = float(ref.F / std::pow(s_ref, nexp));
		} else if (have && !fig.empty && s0 > 0.0) {
			const double nexp = MAX(fig.n_exp, 1.0);
			r_patch.mLawExponent = float(nexp);
			r_patch.mLawCoefficient = float(fig.F / std::pow(MAX(d_deep, 1e-9), nexp));
		} else if (two_layers) {
			// Two soft sides across a gap: the series law the pressed patch uses (the two-sided design outside the table,
			// with its face ratio), read at the squash where the contact's half-width reaches the softer layer's depth,
			// sqrt(2 R s) = h. The grid cannot resolve a contact much narrower than its cells (N_SHORT cells across a box
			// of half-width sqrt(2 R s) + h): at a thousandth of the depth it finds no contact at all.
			const double s_two = MAX(1e-3 * co.h, co.h * co.h / (2.0 * R));
			CurvedFig ref2;
			if (curved_outside(cls == 1, s_two, R, co, 0.0, ya, yb, ref2) && !ref2.empty && ref2.F > 0.0) {
				const double ratio2 = table_ratio(false, shape_factor(ref2.area, ref2.Ixx, ref2.Iyy, soft.thickness), mu, kg_of(soft));
				const double nexp = MAX(ref2.n_exp, 1.0);
				r_patch.mLawExponent = float(nexp);
				r_patch.mLawCoefficient = float(ref2.F * ratio2 / std::pow(s_two, nexp));
			}
		}
		// The first-touch law the solver reads (Stage 1c-b): the pressed law itself at six squashes, the same law this
		// patch uses when it presses. In the table, from a thousandth of the depth to the table's last squash (0.316 of
		// it), the level line's slices or the point's figures; outside the table, for two soft sides, or where one
		// lookup fails, the pressed fallback (the split law on the grid with its face ratio) from where the grid resolves
		// the contact (a thousandth of the depth, or a quarter of the squash at which the contact's half-width reaches the
		// depth, h^2 / 8R) to half the layers' whole depth. (Two curved soft sides read the two-sided design's law here as
		// when pressed, not the lubricated series of the lead's decision 5, which the table does not cover: disclosed.)
		const bool pressing = have && !fig.empty && fig.F > 0.0;
		if (onset_pending || !pressing) {
			int kept = 0;
			if (!two_layers) {
				kept = set_law_samples(1e-3 * soft.thickness, 0.316 * soft.thickness, 6, true, [&](double p_s, double &r_F) {
					CurvedFig f;
					const bool ok = cls == 1 ? table_line(p_s, 0.0, ya, yb, R, soft, mu, f) : table_point(p_s, R, soft, mu, f);
					r_F = f.F;
					return ok && !f.empty;
				}, r_patch);
			}
			if (kept == 0) {
				const double hh = co.h;
				set_law_samples(MAX(1e-3 * hh, 0.25 * hh * hh / (2.0 * R)), 0.5 * layer, 6, false, [&](double p_s, double &r_F) {
					CurvedFig f;
					if (!curved_outside(cls == 1, p_s, R, co, 0.0, ya, yb, f) || f.empty || !(f.F > 0.0)) {
						return false;
					}
					r_F = f.F * table_ratio(false, shape_factor(f.area, f.Ixx, f.Iyy, soft.thickness), mu, kg_of(soft));
					return true;
				}, r_patch);
			}
		}
	} else if (soft.kind == JoltSoftSurface3D::KIND_SOLID) {
		const double Es = e_star(soft, partner);
		layer = 1e6;
		if (cls == 2) {
			if (s0 > 0.0) {
				const double a = std::sqrt(R * s0);
				fig.F = 4.0 / 3.0 * Es * std::sqrt(R) * std::pow(s0, 1.5);
				fig.r_s = 3.0 * PI * a / 16.0;
				fig.m1x = fig.m1y = 3.0 * a / 8.0;
				fig.k_n = 2.0 * Es * a;
				fig.area = PI * a * a;
				fig.J_p = 0.5 * PI * a * a * a * a;
				fig.sigma = a;
				fig.n_exp = 1.5;
				fig.empty = !(fig.F > 0.0);
				have = true;
			}
			r_patch.mLawCoefficient = float(4.0 / 3.0 * Es * std::sqrt(R));
			r_patch.mLawExponent = 1.5f;
			// The first-touch law the solver reads: Hertz itself (two samples: one segment of exponent 1.5)
			if (onset_pending || !(s0 > 0.0)) {
				set_law_samples(1e-4 * R, 0.03 * R, 2, true, [&](double p_s, double &r_F) {
					r_F = 4.0 / 3.0 * Es * std::sqrt(R) * std::pow(p_s, 1.5);
					return true;
				}, r_patch);
			}
		} else {
			// UNSOURCED line law (design 2.3): slices along the line
			const double dy = (yb - ya) / N_SLICES;
			double F = 0.0, Fy = 0.0, area = 0.0, m1 = 0.0, bmax = 0.0, kn = 0.0;
			double P[N_SLICES], ys[N_SLICES], kslice[N_SLICES];
			for (int i = 0; i < N_SLICES; ++i) {
				ys[i] = ya + (i + 0.5) * dy;
				const double s = s0 + camber * ys[i];
				P[i] = 0.0;
				kslice[i] = 0.0;
				if (s > 0.0) {
					double b;
					solid_line_load(s, R, Es, P[i], b);
					double P2, b2;
					solid_line_load(s * 1.001, R, Es, P2, b2);
					kslice[i] = (P2 - P[i]) / (0.001 * s);
					kn += kslice[i] * dy;
					F += P[i] * dy;
					Fy += P[i] * ys[i] * dy;
					area += 2.0 * b * dy;
					m1 += P[i] * 4.0 * b / (3.0 * PI) * dy;
					bmax = MAX(bmax, b);
				}
			}
			if (F > 0.0) {
				fig.F = F;
				fig.cy = Fy / F;
				double rs = 0.0;
				// The camber stiffness about the rolling direction through the centre, from each slice's stiffness, and
				// the loaded ends (the tilt row's arms), as the layer table's strip model gives them
				double ktx = 0.0;
				bool first = true;
				for (int i = 0; i < N_SLICES; ++i) {
					rs += P[i] * std::abs(ys[i] - fig.cy) * dy;
					ktx += kslice[i] * (ys[i] - fig.cy) * (ys[i] - fig.cy) * dy;
					if (P[i] > 0.0) {
						if (first) {
							fig.y_lo = ys[i] - 0.5 * dy;
							first = false;
						}
						fig.y_hi = ys[i] + 0.5 * dy;
					}
				}
				fig.kt_x = ktx;
				fig.r_s = rs / F;
				fig.m1x = m1 / F;
				fig.area = area;
				fig.sigma = bmax;
				fig.k_n = kn;
				fig.n_exp = 1.0;
				fig.empty = false;
				have = true;
			}
			r_patch.mLawCoefficient = float(MAX(kn, 0.0));
			r_patch.mLawExponent = 1.0f;
			// The first-touch law the solver reads (Stage 1c-b): the level line's slices at six squashes from 1e-4 R to
			// 0.03 R; across a gap the slices carry nothing, so without them no first touch could be decided
			if (onset_pending || !(F > 0.0)) {
				const double L = yb - ya;
				set_law_samples(1e-4 * R, 0.03 * R, 6, true, [&](double p_s, double &r_F) {
					double P, b;
					solid_line_load(p_s, R, Es, P, b);
					r_F = L * P;
					return true;
				}, r_patch);
			}
		}
	} else {
		// Lumped: F = K_l s with the class's bed-of-springs profile. At a point, the squash s0. Along a line, a bed of
		// springs of K_l / L per length over the loaded length L, each slice at its own squash: camber moves the centre
		// of pressure, and the line resists it (its camber stiffness and loaded ends feed the tilt row)
		layer = soft.travel > 0.0 ? soft.travel : 1e6;
		if (cls == 2) {
			if (s0 > 0.0) {
				const double a = std::sqrt(2.0 * R * s0);
				fig.F = soft.lumped_stiffness * s0;
				fig.k_n = soft.lumped_stiffness;
				fig.n_exp = 1.0;
				fig.sigma = a;
				fig.m1x = fig.m1y = 16.0 * a / (15.0 * PI);
				fig.r_s = 8.0 * a / 15.0;
				fig.area = PI * a * a;
				fig.empty = false;
				have = true;
			}
		} else if (d_deep > 0.0) {
			const double L = yb - ya;
			const double kl = soft.lumped_stiffness / L;
			const double dy = L / N_SLICES;
			double F = 0.0, Fy = 0.0, kn = 0.0, len = 0.0;
			double P[N_SLICES], ys[N_SLICES];
			for (int i = 0; i < N_SLICES; ++i) {
				ys[i] = ya + (i + 0.5) * dy;
				const double s = s0 + camber * ys[i];
				P[i] = s > 0.0 ? kl * s : 0.0;
				if (s > 0.0) {
					F += P[i] * dy;
					Fy += P[i] * ys[i] * dy;
					kn += kl * dy;
					len += dy;
				}
			}
			if (F > 0.0) {
				fig.F = F;
				fig.cy = Fy / F;
				fig.k_n = kn;
				double rs = 0.0, ktx = 0.0;
				bool first = true;
				for (int i = 0; i < N_SLICES; ++i) {
					if (P[i] > 0.0) {
						rs += P[i] * std::abs(ys[i] - fig.cy) * dy;
						ktx += kl * (ys[i] - fig.cy) * (ys[i] - fig.cy) * dy;
						if (first) {
							fig.y_lo = ys[i] - 0.5 * dy;
							first = false;
						}
						fig.y_hi = ys[i] + 0.5 * dy;
					}
				}
				const double a = std::sqrt(2.0 * R * MAX(s0 + camber * fig.cy, 0.0));
				fig.m1x = 3.0 * a / 8.0;
				fig.r_s = rs / F;
				fig.kt_x = ktx;
				fig.area = 2.0 * a * len;
				fig.sigma = a;
				fig.n_exp = 1.0;
				fig.empty = false;
				have = true;
			}
		}
		r_patch.mLawCoefficient = float(soft.lumped_stiffness);
		r_patch.mLawExponent = 1.0f;
		if (onset_pending || !have) {
			set_law_linear(soft.lumped_stiffness, MIN(layer, R), r_patch);
		}
	}

	r_patch.mRigidStick = rigid_stick;
	r_patch.mCoreGap = float(layer - d_deep);
	// The core guard: a Line's two loaded ends, a Point's deepest point
	if (cls == 1) {
		const double ends[2] = { ya, yb };
		for (int i = 0; i < 2; ++i) {
			const double d = s0 + camber * ends[i];
			r_patch.mCorePoint[i] = line_world(0.0, ends[i]);
			r_patch.mCoreSquash[i] = float(d);
			r_patch.mCoreGapAt[i] = float(layer - d);
		}
		r_patch.mNumCores = 2;
	} else {
		r_patch.mCorePoint[0] = r_patch.mDeepest;
		r_patch.mCoreSquash[0] = float(d_deep);
		r_patch.mCoreGapAt[0] = float(layer - d_deep);
		r_patch.mNumCores = 1;
	}
	r_patch.mRollingRadius = float(R);
	r_result.face_ratio = fig.face_ratio;
	if (have && !fig.empty && fig.F > 0.0) {
		r_result.raw_force = fig.F / fig.face_ratio;
		r_patch.mForce = float(fig.F);
		r_patch.mNormalStiffness = float(MAX(fig.k_n, 0.0));
		r_patch.mNormalDampingRatio = float(loss_side.zeta_for(fig.n_exp));
		r_patch.mCentre = line_world(fig.cx, fig.cy);
		r_patch.mSpinRadius = float(fig.r_s);
		r_patch.mArea = float(fig.area);
		r_patch.mRelaxationLength = float(fig.sigma);
		r_result.m1[0] = fig.m1x;
		r_result.m1[1] = fig.m1y;
		if (!rigid_stick) {
			r_patch.mShearStiffness = float(shear_G * fig.area / soft.thickness);
			r_patch.mSpinStiffness = float(shear_G * fig.J_p / soft.thickness);
		}
		if (cls == 1) {
			// Camber: tilt about the rolling direction t1, the moment held by the loaded length on each side; rolling
			// resistance about the line, its arm (alpha / 2) m1 across the line plus the measured one
			r_patch.mNumTilt = 1;
			r_patch.mTiltAxis[0] = t1;
			r_patch.mTiltStiffness[0] = float(MAX(fig.kt_x, 0.0));
			r_patch.mTiltArmPos[0] = float(MAX(fig.y_hi - fig.cy, 0.0));
			r_patch.mTiltArmNeg[0] = float(MAX(fig.cy - fig.y_lo, 0.0));
			r_patch.mNumRoll = 1;
			r_patch.mRollAxis[0] = t2;
			r_patch.mRollArm[0] = float(rolling_arm(fig.m1x, loss_side.loss, additive_arm));
		} else {
			// A point rolls about both in-plane axes
			r_patch.mNumTilt = 0;
			r_patch.mNumRoll = 2;
			r_patch.mRollAxis[0] = t1;
			r_patch.mRollAxis[1] = t2;
			r_patch.mRollArm[0] = float(rolling_arm(fig.m1y, loss_side.loss, additive_arm));
			r_patch.mRollArm[1] = float(rolling_arm(fig.m1x, loss_side.loss, additive_arm));
		}
	} else {
		// Nothing presses yet: no force, the rows act on the predicted overlap. A Point's is at its deepest point. A
		// Line's spreads along it, so its row acts at the centroid of the overlap the line would have one speculative
		// band (w) deeper than now, d(y) - d_deep + w: a level line at its middle, a cambered one toward the end that
		// touches first (at the deeper end itself, a level line's row would turn it about the far end)
		r_patch.mForce = 0.0f;
		r_patch.mNormalStiffness = 0.0f;
		r_patch.mNormalDampingRatio = float(loss_side.zeta_for(MAX(double(r_patch.mLawExponent), 1.0)));
		r_patch.mCentre = cls == 1 ? line_world(0.0, y_band) : r_patch.mDeepest;
	}
	r_patch.mEnabled = true;
	r_result.outcome = JoltSoftBuildResult::BUILT;
	return true;
}

void jolt_soft_patch_debug_figures(const JPH::SoftContactPatch &p_patch, const JPH::Vec3 &p_normal, const JoltSoftBuildResult &p_result, int p_kind, float *r_figures) {
	float *w = r_figures;
	for (int i = 0; i < JOLT_SOFT_PATCH_FIGURES; ++i) {
		w[i] = 0.0f;
	}
	w[0] = float(p_patch.mClass);
	w[1] = float(p_kind);
	w[2] = p_patch.mForce;
	w[3] = p_patch.mSquash;
	w[4] = p_patch.mNormalStiffness;
	w[5] = float(p_patch.mCentre.GetX());
	w[6] = float(p_patch.mCentre.GetY());
	w[7] = float(p_patch.mCentre.GetZ());
	w[8] = p_normal.GetX();
	w[9] = p_normal.GetY();
	w[10] = p_normal.GetZ();
	w[11] = p_patch.mNumTilt > 0 ? p_patch.mTiltStiffness[0] : 0.0f;
	w[12] = p_patch.mNumTilt > 1 ? p_patch.mTiltStiffness[1] : 0.0f;
	w[13] = p_patch.mSpinRadius;
	w[14] = p_patch.mNumRoll > 0 ? p_patch.mRollArm[0] : 0.0f;
	w[15] = p_patch.mNumRoll > 1 ? p_patch.mRollArm[1] : 0.0f;
	w[16] = p_patch.mArea;
	w[17] = float(p_result.m1[0]);
	w[18] = float(p_result.m1[1]);
	w[19] = p_patch.mStaticFriction;
	w[20] = p_patch.mSlidingFriction;
	w[21] = float(p_result.face_ratio);
	w[22] = p_result.outside_table ? 1.0f : 0.0f;
	w[23] = float(p_result.passes);
	w[24] = p_result.not_settled ? 0.0f : 1.0f;
	w[25] = p_patch.mShearStiffness;
	w[26] = p_patch.mSpinStiffness;
	w[27] = p_patch.mRestitution;
	w[28] = p_patch.mLawCoefficient;
	w[29] = p_patch.mLawExponent;
	w[30] = p_patch.mTangent1.GetX();
	w[31] = p_patch.mTangent1.GetY();
	w[32] = p_patch.mTangent1.GetZ();
	w[33] = float(p_result.raw_force);
	w[34] = p_patch.mCoreGap;
	w[35] = float(p_patch.mDeepest.GetX());
	w[36] = float(p_patch.mDeepest.GetY());
	w[37] = float(p_patch.mDeepest.GetZ());
	w[38] = p_patch.mRelaxationLength;
	w[39] = p_patch.mRigidStick ? 1.0f : 0.0f;
}
