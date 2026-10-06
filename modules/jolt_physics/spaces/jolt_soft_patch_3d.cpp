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

#include "soft_layer_table/jolt_soft_face_table.gen.h"
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
#include <cstring>
#include <limits>

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

// The vertices p_src uses into r_dst (no reader of a Poly looks past its n, and a whole Poly is 1160 bytes)
void copy_poly(const Poly &p_src, Poly &r_dst) {
	r_dst.n = p_src.n;
	memcpy(r_dst.x, p_src.x, sizeof(double) * p_src.n);
	memcpy(r_dst.y, p_src.y, sizeof(double) * p_src.n);
}

// p_poly clipped by the convex p_clip (both made counter-clockwise first). The edges clip into two buffers in turn,
// each copy holding only the vertices in use: the face law clips twice a grid cell.
void clip_convex(const Poly &p_poly, const Poly &p_clip, Poly &r_out) {
	Poly buf[2];
	copy_poly(p_poly, buf[0]);
	make_ccw(buf[0]);
	Poly clip;
	copy_poly(p_clip, clip);
	make_ccw(clip);
	int cur = 0;
	for (int k = 0; k < clip.n; ++k) {
		if (buf[cur].n == 0) {
			break;
		}
		const int k1 = (k + 1) % clip.n;
		const double a = -(clip.y[k1] - clip.y[k]);
		const double b = clip.x[k1] - clip.x[k];
		const double c = -(a * clip.x[k] + b * clip.y[k]);
		clip_halfplane(buf[cur], a, b, c, buf[cur ^ 1]);
		cur ^= 1;
	}
	copy_poly(buf[cur], r_out);
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
		if (p_tail != nullptr && p_tail->mGridKind == JPH::SoftContactTail::GridSplit && make_grid(bx0, bx1, by0, by1, 0, N_SHORT, probe) && p_tail->mGridLong == probe.nl && p_tail->mGridShort == probe.ns) {
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

// ---------------------------------------------------------------------------------------------------------------
// The face law v2 (Stage 1c-b desk A, DESK_A.md 2 to 6; the desk reference's section 6b): a Layer face against a rigid
// partner. The layer's film flow in conservative form on a tensor grid of cells: p = alpha (E/h) w + q,
// q / kq + div Q = w, Q = -kappa grad q (plus the held face's Couette term (h/2) u_s), kappa h^3 / (12 G) under a face
// that holds and h^3 / (3 G) under a frictionless face and over the free layer beyond the contact, w = -(h / (gamma E)) q
// outside the contact and q = 0 at the grid's edge; the contact set by complementarity. Two fields: a frictionless face
// ('lub') and the physical no-slip face ('ns'), which holds each point where it touched (a stateless model of the slide
// it made while still free). An indenter edge (the partner's face ending inside the layer) adds the resistance of a held
// film CHOKE depths long. The FE's edge multipliers act cell by cell from the nearest free face and the nearest indenter
// edge, the Coulomb face is p = (1 - Phi) p_lub + Phi p_ns with Phi the FE's own friction share along the contact's
// boundary, then the FE's own reshaping at mu 0.3, 0.6 and 0.9. The FE tables are the face table header's
// (soft_layer_table/jolt_soft_face_table.gen.h); its constants and stations must be this code's (checked below).
// The face law v3 (Stage 1c-b fix round; the desk reference's section 6c, its inset tables from
// stage1cb/fx/gen_inset_tables.py) is v2 with:
//   1. an indenter edge by its margin m (the free layer between the partner's edge and the layer's own edge, read at
//      the edge's point nearest the cell, at most FREE_BOX depths): its multipliers and friction share from the inset
//      tables (FE by margin: 0 is the free face, FREE_BOX the punch), its choke times chi = 1 - exp(-m / INSET_LAMBDA);
//   2. a cell near both a free face and an indenter edge of small margin takes them as one free face (a soft choice by
//      distance, weighted by the indenter's freeness 1 - chi), so a part's edge crossing the pad's reads continuously;
//   3. the grid turned to the circular mean (modulo 90 deg) of the contactable region's near-longest edges, the longest
//      edge when they disagree, so a part turning on a pad its own size turns the grid continuously;
//   4. embedded edges: a cell keeps its centre's classification but stands for the area the faces really cover (its own
//      covered part plus its axis neighbours' whose centres fall outside), q = 0 where each axis of a cell meets the
//      layer's edge, an indenter edge where the partner's edge crosses the axis between two cells; on an axis-aligned
//      rectangle every cell is whole and the discretisation is v2's;
//   5. at least 12 cells per axis (a round contact's edge on the embedded grid);
//   6. the held slide's memory: the gradient remembered as the contact set grew (area weighted, carried in the tail) in
//      place of the current gradient, so a face pressed flat and then tilted makes no slide.

namespace ftab = jolt_soft_face_table;

constexpr double F2_A_HELD = 12.0; // film coefficient under a held face, kappa = h^3 / (A G)
constexpr double F2_A_FREE = 3.0; // under a frictionless face and over the free layer
constexpr double F2_ALPHA_LUB = 1.8; // the local term alpha (E/h) of the frictionless field
constexpr double F2_ALPHA_NS = 1.0; // and of the no-slip field
constexpr double F2_GAMMA_LUB = 2.0; // outside the contact w = -(h / (gamma E)) q
constexpr double F2_GAMMA_NS = 2.0;
constexpr double F2_CHOKE = 0.2; // an indenter edge's resistance, in depths of held film
constexpr double F2_KS = 0.3; // the held slide: u_s = KS |g| s (s / h)^(KP - 1) along -g
constexpr double F2_KP = 1.25;
constexpr double F2_BOX = 4.0; // the free layer kept beyond the contactable region, in depths
constexpr double F2_CELL_H = 0.35; // inner cells at most this many depths across where the count allows
constexpr int F2_N_MIN = 12; // inner cells per axis, at least (v3: the edge of a round contact needs them)
constexpr int F2_N_MAX = 32; // and at most (graded from CELL_H depths at both ends beyond it)
constexpr int F2_N_OUT_MAX = 6; // outer cells each side, at most
constexpr double F2_OUT_RATIO = 1.5; // their growth
constexpr int F2_MAX_CELLS = 1024; // the whole grid, at most
constexpr int F2_CAP_PASSES = 8; // active-set passes
constexpr int F2_MAX_LINES = 64; // grid lines per axis (inner 33, outer 6 each side)
constexpr int F2_MAX_BAND = 33; // the half-bandwidth (the shorter axis's cells, at most 32) plus one
// A partner edge within this many depths inside the layer's own edge is that free face, and a layer edge within it
// outside the partner is still pressed to its end: the engine's faces carry float noise, and an exact coincidence (a
// block the size of its pad) is the FE's free face, not a punch
constexpr double F2_EDGE_TOL = 1e-3;
constexpr int F2_NUM_E = 21;
constexpr double F2_STATIONS[F2_NUM_E] = { 0.0, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0 };
constexpr int F2_TAIL_WORDS = F2_MAX_CELLS / 32; // one field's contact set in the tail
constexpr double F3_SOFTMIN = 0.05; // the soft choice between a free face and an indenter edge, depths
constexpr double F3_ALIGN_EPS = 0.1; // an edge's weight in the grid's turn: L exp(-(Lmax - L) / (EPS Lmax))
constexpr double F3_ALIGN_AGREE = 0.5; // the weighted mean's length that makes it the turn (else the longest edge)
constexpr double F3_GHOST_MIN = 0.05; // a drain at least this fraction of the cell from its centre
constexpr double F3_IND_SPAN = 0.02; // an indenter edge at least this fraction of the centres' span from either centre
constexpr double F3_INSET_LAMBDA = 0.75; // the choke's growth with the margin (the inset tables were made with it)

constexpr bool f2_stations_match() {
	for (int i = 0; i < F2_NUM_E; ++i) {
		if (ftab::STATION[i] != float(F2_STATIONS[i])) {
			return false;
		}
	}
	return true;
}

static_assert(ftab::FACE_VERSION == 2, "The module reads version 2 of the face table (the inset tables, stage1cb/fx/make_face_header_v2.py).");
static_assert(ftab::INSET_LAMBDA == float(F3_INSET_LAMBDA) && ftab::NUM_M >= 2, "The face table's inset tables were made for another choke growth.");
static_assert(ftab::A_HELD == float(F2_A_HELD) && ftab::A_FREE == float(F2_A_FREE), "The face table was made for other film coefficients.");
static_assert(ftab::ALPHA_LUB == float(F2_ALPHA_LUB) && ftab::ALPHA_NS == float(F2_ALPHA_NS), "The face table was made for other local terms.");
static_assert(ftab::GAMMA_LUB == float(F2_GAMMA_LUB) && ftab::GAMMA_NS == float(F2_GAMMA_NS), "The face table was made for another free-region compliance.");
static_assert(ftab::CHOKE == float(F2_CHOKE) && ftab::HELD_KS == float(F2_KS) && ftab::HELD_KP == float(F2_KP), "The face table was made for another indenter edge or held slide.");
static_assert(ftab::FREE_BOX == float(F2_BOX), "The face table was made for another free layer box.");
static_assert(ftab::NUM_E == F2_NUM_E && ftab::NUM_Q == F2_NUM_E + 1 && f2_stations_match(), "The face table's stations are not this law's.");
static_assert(ftab::NUM_MIX == ftab::NUM_MU - 2 && ftab::NUM_GEO == 2 && ftab::NUM_TAU == 2, "The face table's layout is not the one this code reads.");
static_assert(F2_TAIL_WORDS * 2 == JPH::SoftContactTail::cActiveSetWords, "The tail holds the two fields' contact sets.");

// The reference rounds every grid line to 12 decimals and keeps the distinct ones in order
double f2_round12(double p_v) {
	return std::nearbyint(p_v * 1e12) / 1e12;
}

int f2_unique(double *r_v, int p_n) {
	for (int i = 0; i < p_n; ++i) {
		r_v[i] = f2_round12(r_v[i]);
	}
	for (int i = 1; i < p_n; ++i) {
		const double v = r_v[i];
		int j = i - 1;
		while (j >= 0 && r_v[j] > v) {
			r_v[j + 1] = r_v[j];
			--j;
		}
		r_v[j + 1] = v;
	}
	int m = 0;
	for (int i = 0; i < p_n; ++i) {
		if (m == 0 || r_v[i] != r_v[m - 1]) {
			r_v[m++] = r_v[i];
		}
	}
	return m;
}

// A sum in the reference's order (pairwise, eight accumulators from eight values on)
double f2_pairwise_sum(const double *p_a, int p_n) {
	if (p_n < 8) {
		double r = 0.0;
		for (int i = 0; i < p_n; ++i) {
			r += p_a[i];
		}
		return r;
	}
	double r[8];
	for (int j = 0; j < 8; ++j) {
		r[j] = p_a[j];
	}
	int i = 8;
	for (; i < p_n - (p_n % 8); i += 8) {
		for (int j = 0; j < 8; ++j) {
			r[j] += p_a[i + j];
		}
	}
	double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
	for (; i < p_n; ++i) {
		res += p_a[i];
	}
	return res;
}

// The contactable span's lines: n = clamp(ceil(L / (CELL_H h)), N_MIN, N_MAX) cells, made even; equal when that holds
// CELL_H h, else N_MAX cells graded geometrically from CELL_H h at both ends. p_n_cap > 0 lowers N_MAX (the cell cap).
int f2_inner_lines(double p_a, double p_b, double p_h, int p_n_cap, double *r_lines) {
	const double L = p_b - p_a;
	const int nmax = p_n_cap > 0 ? MAX(MIN(p_n_cap, F2_N_MAX), F2_N_MIN) : F2_N_MAX;
	int n = int(std::ceil(L / (F2_CELL_H * p_h) - 1e-9));
	n = MIN(MAX(n, F2_N_MIN), nmax);
	n += n % 2;
	const double s0 = F2_CELL_H * p_h;
	double v[2 * F2_MAX_LINES];
	int count = 0;
	if (L / n <= s0 * (1.0 + 1e-9)) {
		const double step = L / n;
		for (int i = 0; i < n; ++i) {
			v[count++] = double(i) * step + p_a;
		}
		v[count++] = p_b;
		memcpy(r_lines, v, sizeof(double) * count);
		return f2_unique(r_lines, count);
	}
	const int m = n / 2;
	double lo = 1.0, hi = 4.0;
	for (int it = 0; it < 60; ++it) {
		const double r = 0.5 * (lo + hi);
		const double tot = 2.0 * s0 * (std::pow(r, double(m)) - 1.0) / (r - 1.0);
		if (tot > L) {
			hi = r;
		} else {
			lo = r;
		}
	}
	const double r = 0.5 * (lo + hi);
	double sizes[F2_N_MAX];
	for (int k = 0; k < m; ++k) {
		sizes[k] = s0 * std::pow(r, double(k));
	}
	const double scale = (L / 2.0) / f2_pairwise_sum(sizes, m);
	double half_edges[F2_N_MAX + 1];
	half_edges[0] = 0.0;
	double acc = 0.0;
	for (int k = 0; k < m; ++k) {
		sizes[k] *= scale;
		acc += sizes[k];
		half_edges[k + 1] = acc;
	}
	for (int k = 0; k <= m; ++k) {
		v[count++] = p_a + half_edges[k];
	}
	// The two halves meet at one middle line, the first half's. The second half's own middle, p_b less the same sum, can
	// differ from it by a few ulps, and when the two fall either side of a rounding step of f2_unique they are kept as two
	// lines 1e-12 apart: a sliver cell, 2 m + 1 cells in all. The cell cap's pass then asks for two cells fewer, the count
	// is made even again, the same lines come back and the step never ends (the maze robot's soft tyre at a wall).
	for (int k = 1; k <= m; ++k) {
		v[count++] = p_b - half_edges[m - k];
	}
	memcpy(r_lines, v, sizeof(double) * count);
	return f2_unique(r_lines, count);
}

// At most N_OUT_MAX cells from the inner edge out to p_end, growing by OUT_RATIO from p_dx0, the last stretched
int f2_outer_lines(double p_edge, double p_end, double p_dx0, double *r_lines) {
	const double span = std::abs(p_end - p_edge);
	const double sgn = p_end >= p_edge ? 1.0 : -1.0;
	double out[F2_N_OUT_MAX];
	int count = 0;
	double s = 0.0;
	double dx = p_dx0 * F2_OUT_RATIO;
	while (span - s > 1e-12 * MAX(1.0, span) && count < F2_N_OUT_MAX) {
		if (s + dx >= span || span - (s + dx) < 0.5 * dx * F2_OUT_RATIO || count == F2_N_OUT_MAX - 1) {
			out[count++] = span;
			s = span;
			break;
		}
		s += dx;
		out[count++] = s;
		dx *= F2_OUT_RATIO;
	}
	if (count > 0 && std::abs(out[count - 1] - span) > 1e-12 * MAX(1.0, span)) {
		out[count - 1] = span;
	}
	for (int k = 0; k < count; ++k) {
		r_lines[k] = p_edge + sgn * out[k];
	}
	return count;
}

// One axis's lines: the inner span [a, b] and the free layer out to [lo, hi]
int f2_engine_lines(double p_lo, double p_a, double p_b, double p_hi, double p_h, int p_n_cap, double *r_lines) {
	const double a = MAX(p_a, p_lo);
	const double b = MIN(p_b, p_hi);
	double inner[2 * F2_MAX_LINES];
	const int ni = f2_inner_lines(a, b, p_h, p_n_cap, inner);
	if (ni < 2) {
		return 0;
	}
	double left[F2_N_OUT_MAX], right[F2_N_OUT_MAX];
	const int nl = f2_outer_lines(a, p_lo, inner[1] - inner[0], left);
	const int nr = f2_outer_lines(b, p_hi, inner[ni - 1] - inner[ni - 2], right);
	double v[2 * F2_MAX_LINES + 2 * F2_N_OUT_MAX];
	int count = 0;
	for (int k = nl - 1; k >= 0; --k) {
		v[count++] = left[k];
	}
	for (int k = 0; k < ni; ++k) {
		v[count++] = inner[k];
	}
	for (int k = 0; k < nr; ++k) {
		v[count++] = right[k];
	}
	count = f2_unique(v, count);
	if (count > F2_MAX_LINES) {
		return 0;
	}
	memcpy(r_lines, v, sizeof(double) * count);
	return count;
}

// The tensor grid: nx by ny cells over the lines xe, ye, numbered with the shorter axis fastest so the half-bandwidth
// is the shorter axis's count
struct F2Grid {
	int nx = 0;
	int ny = 0;
	int n = 0;
	int m = 0;
	bool y_fast = true;
	double xe[F2_MAX_LINES];
	double ye[F2_MAX_LINES];

	int k(int p_i, int p_j) const { return y_fast ? p_i * ny + p_j : p_j * nx + p_i; }
	int ix(int p_k) const { return y_fast ? p_k / ny : p_k % nx; }
	int jy(int p_k) const { return y_fast ? p_k % ny : p_k / nx; }
};

// The edge stretch a -> b clipped to the convex counter-clockwise p_ccw grown outward by p_grow (shrunk when negative):
// the parameter interval [t0, t1] of a -> b inside it (Cyrus and Beck), false when none
bool f2_clip_segment(double p_ax, double p_ay, double p_bx, double p_by, const Poly &p_ccw, double p_grow, double &r_t0, double &r_t1) {
	double t0 = 0.0, t1 = 1.0;
	const double dx = p_bx - p_ax, dy = p_by - p_ay;
	for (int k = 0; k < p_ccw.n; ++k) {
		const int k1 = (k + 1) % p_ccw.n;
		const double ex = p_ccw.x[k1] - p_ccw.x[k];
		const double ey = p_ccw.y[k1] - p_ccw.y[k];
		const double fa = ex * (p_ay - p_ccw.y[k]) - ey * (p_ax - p_ccw.x[k]) + (p_grow != 0.0 ? p_grow * std::sqrt(ex * ex + ey * ey) : 0.0);
		const double fd = ex * dy - ey * dx;
		if (std::abs(fd) < 1e-15) {
			if (fa < -1e-9 * MAX(1.0, std::abs(ex) + std::abs(ey))) {
				return false;
			}
			continue;
		}
		const double t = -fa / fd;
		if (fd > 0.0) {
			t0 = MAX(t0, t);
		} else {
			t1 = MIN(t1, t);
		}
		if (t0 > t1 + 1e-12) {
			return false;
		}
	}
	r_t0 = t0;
	r_t1 = t1;
	return t1 - t0 > 1e-9;
}

// One field: its contact set, the pressure of its last solve, and the tangent pressures (alpha (E/h) w + q on the held
// set for the test displacements w = 1, X and Y)
struct F2Field {
	bool solved = false;
	bool any = false;
	bool settled = false;
	int passes = 0;
	bool active[F2_MAX_CELLS];
	double p[F2_MAX_CELLS];
	double t1[F2_MAX_CELLS];
	double tx[F2_MAX_CELLS];
	double ty[F2_MAX_CELLS];
};

// The law's figures in the caller's frame, with the grid and both fields kept for the tilt arms and the tail
struct F2Law {
	bool empty = true;
	double F = 0.0, cx = 0.0, cy = 0.0, r_s = 0.0, m1x = 0.0, m1y = 0.0, area = 0.0, Ixx = 0.0, Iyy = 0.0, J_p = 0.0;
	double k_n = 0.0, kt_x = 0.0, kt_y = 0.0, kt_xy = 0.0;
	double edge_x[2] = { 0.0, 0.0 }, edge_y[2] = { 0.0, 0.0 };
	double Phi = 0.0;
	double wts[2] = { 0.0, 0.0 }; // the lub and ns fields' weights
	double rot = 0.0; // the grid's axes are the caller's turned by rot
	int passes = 0;
	bool settled = true;
	double A_act = 0.0; // the no-slip contact's area (the areas its cells stand for)
	double held_out[3] = { 0.0, 0.0, 0.0 }; // the held slide's memory after this evaluation: gradient x, y (the caller's frame), area
	F2Grid grid;
	double X[F2_MAX_CELLS], Y[F2_MAX_CELLS], DX[F2_MAX_CELLS], DY[F2_MAX_CELLS]; // in the grid's frame
	F2Field field[2]; // 0 lub, 1 ns
};

// A field's last-step contact sets, as the tail keeps them
struct F2Tail {
	const uint32_t *words = nullptr; // the lub field's F2_TAIL_WORDS, then the ns field's
	int nx = 0;
	int ny = 0;
	bool has[2] = { false, false };
	bool has_held = false; // the held slide's memory (rule 6): gradient x, y in the caller's frame, and the area
	double held[3] = { 0.0, 0.0, 0.0 };
};

// The working arrays of one evaluation (the grid's cells, the edge distances, the system)
struct F2Work {
	double area[F2_MAX_CELLS];
	bool dom[F2_MAX_CELLS];
	bool cont[F2_MAX_CELLS];
	double d[F2_MAX_CELLS];
	double hx[F2_MAX_CELLS], hy[F2_MAX_CELLS];
	double e[2][F2_MAX_CELLS], ext[2][F2_MAX_CELLS], anx[2][F2_MAX_CELLS], any[2][F2_MAX_CELLS]; // [0] free face, [1] indenter edge
	double px[F2_MAX_CELLS], py[F2_MAX_CELLS]; // the nearest indenter edge's nearest point
	double marg[F2_MAX_CELLS], chi[F2_MAX_CELLS]; // that edge's margin (depths, at most FREE_BOX) and the choke's share
	double stair_ind[F2_MAX_CELLS];
	double Ad[F2_MAX_CELLS], Ac[F2_MAX_CELLS]; // the areas a cell stands for: in the layer's face, under the partner
	bool recv_d[F2_MAX_CELLS], recv_c[F2_MAX_CELLS]; // a cell standing for area up to the face's edge, the partner's
	double dS[F2_MAX_CELLS][4], dP[F2_MAX_CELLS][4]; // where each axis (-x, +x, -y, +y) meets the face's edge, the partner's
	double kap[F2_MAX_CELLS];
	double diag[F2_MAX_CELLS], off_f[F2_MAX_CELLS], off_s[F2_MAX_CELLS];
	double band[F2_MAX_CELLS * F2_MAX_BAND];
	double rhs[F2_MAX_CELLS], q[F2_MAX_CELLS], y[F2_MAX_CELLS];
	double rhs1[F2_MAX_CELLS], rhsx[F2_MAX_CELLS], rhsy[F2_MAX_CELLS];
	double y2[F2_MAX_CELLS], y3[F2_MAX_CELLS]; // the tangent solves' second and third forward sweeps
	double M[2][F2_MAX_CELLS], R[F2_MAX_CELLS], pc[F2_MAX_CELLS];
	bool next[F2_MAX_CELLS];
	double E = 0.0, h = 0.0, G = 0.0, Eh = 0.0, kq = 0.0, kg = 0.0;
};

// The film system of one field for a given contact set (symmetric positive definite, banded)
void f2_assemble_matrix(const F2Law &p_law, F2Work &w, const bool *p_active, int p_kind) {
	const F2Grid &g = p_law.grid;
	const double h = w.h;
	const double kb = h * h * h / (F2_A_HELD * w.G);
	const double kf = h * h * h / (F2_A_FREE * w.G);
	const double gam = p_kind == 1 ? F2_GAMMA_NS : F2_GAMMA_LUB;
	const double r_choke = F2_CHOKE * h / kb;
	for (int k = 0; k < g.n; ++k) {
		w.kap[k] = (p_active[k] && p_kind == 1) ? kb : kf;
		w.diag[k] = w.dom[k] ? w.Ad[k] / w.kq : 1.0;
		// the free part of a cell's area carries the free region's compliance
		const double free_area = p_active[k] ? MAX(w.Ad[k] - w.Ac[k], 0.0) : w.Ad[k];
		w.diag[k] += w.dom[k] ? free_area * h / (gam * w.E) : 0.0;
		w.off_f[k] = 0.0;
		w.off_s[k] = 0.0;
	}
	for (int axis = 0; axis < 2; ++axis) {
		const int ni = axis == 0 ? g.nx - 1 : g.nx;
		const int nj = axis == 0 ? g.ny : g.ny - 1;
		for (int i = 0; i < ni; ++i) {
			for (int j = 0; j < nj; ++j) {
				const int a = g.k(i, j);
				const int b = axis == 0 ? g.k(i + 1, j) : g.k(i, j + 1);
				if (!w.dom[a] || !w.dom[b]) {
					continue;
				}
				const bool a_ind = p_active[a] && !w.cont[b];
				const bool b_ind = !a_ind && p_active[b] && !w.cont[a];
				const double la = axis == 0 ? p_law.DX[a] : p_law.DY[a];
				const double lb = axis == 0 ? p_law.DX[b] : p_law.DY[b];
				const double length = axis == 0 ? p_law.DY[a] : p_law.DX[a];
				const double span = 0.5 * la + 0.5 * lb;
				double res;
				if (a_ind) {
					// the partner's edge where it crosses the axis from the pressed cell, its choke there
					const double th = MIN(MAX(w.dP[a][axis == 0 ? 1 : 3], F3_IND_SPAN * span), (1.0 - F3_IND_SPAN) * span);
					res = th / w.kap[a] + (span - th) / w.kap[b] + r_choke * w.stair_ind[a];
				} else if (b_ind) {
					const double th = MIN(MAX(w.dP[b][axis == 0 ? 0 : 2], F3_IND_SPAN * span), (1.0 - F3_IND_SPAN) * span);
					res = th / w.kap[b] + (span - th) / w.kap[a] + r_choke * w.stair_ind[b];
				} else {
					res = 0.5 * la / w.kap[a] + 0.5 * lb / w.kap[b];
				}
				const double cf = length / res;
				if (b == a + 1 && (g.y_fast ? axis == 1 : axis == 0)) {
					w.off_f[a] = -cf;
				} else {
					w.off_s[a] = -cf;
				}
				w.diag[a] += cf;
				w.diag[b] += cf;
			}
		}
	}
	// q = 0 where each axis of a cell meets the layer's edge (at the grid's end, at most half a cell out)
	for (int k = 0; k < g.n; ++k) {
		if (!w.dom[k]) {
			continue;
		}
		const int i = g.ix(k), j = g.jy(k);
		for (int side = 0; side < 4; ++side) {
			const int ni = side == 0 ? i - 1 : (side == 1 ? i + 1 : i);
			const int nj = side == 2 ? j - 1 : (side == 3 ? j + 1 : j);
			const bool inside = ni >= 0 && ni < g.nx && nj >= 0 && nj < g.ny;
			if (inside && w.dom[g.k(ni, nj)]) {
				continue;
			}
			const double la = side < 2 ? p_law.DX[k] : p_law.DY[k];
			const double length = side < 2 ? p_law.DY[k] : p_law.DX[k];
			double dist = w.dS[k][side];
			if (!inside) {
				dist = MIN(dist, 0.5 * la);
			}
			dist = MAX(dist, F3_GHOST_MIN * la);
			w.diag[k] += length * w.kap[k] / dist;
		}
	}
}

// The right-hand side of a field's solve: the interference on the contact set, and the held face's Couette flux
void f2_assemble_rhs(const F2Law &p_law, F2Work &w, const bool *p_active, bool p_held) {
	const F2Grid &g = p_law.grid;
	for (int k = 0; k < g.n; ++k) {
		w.rhs[k] = p_active[k] ? w.d[k] * w.Ac[k] : 0.0;
	}
	if (!p_held) {
		return;
	}
	for (int axis = 0; axis < 2; ++axis) {
		const int ni = axis == 0 ? g.nx - 1 : g.nx;
		const int nj = axis == 0 ? g.ny : g.ny - 1;
		const double *u = axis == 0 ? w.hx : w.hy;
		for (int i = 0; i < ni; ++i) {
			for (int j = 0; j < nj; ++j) {
				const int a = g.k(i, j);
				const int b = axis == 0 ? g.k(i + 1, j) : g.k(i, j + 1);
				if (!w.dom[a] || !w.dom[b]) {
					continue;
				}
				const double us = (p_active[a] && p_active[b]) ? 0.5 * (u[a] + u[b]) : (p_active[a] ? 0.5 * u[a] : (p_active[b] ? 0.5 * u[b] : 0.0));
				const double length = axis == 0 ? p_law.DY[a] : p_law.DX[a];
				const double cq = 0.5 * w.h * us * length;
				w.rhs[a] -= cq;
				w.rhs[b] += cq;
			}
		}
	}
}

// The banded Cholesky factor of the assembled system (fixed order, no pivoting)
bool f2_factor(const F2Grid &g, F2Work &w) {
	const int n = g.n;
	const int m = g.m;
	auto A = [&](int p_i, int p_j) -> double {
		if (p_i == p_j) {
			return w.diag[p_i];
		}
		if (p_i - p_j == m) {
			return w.off_s[p_j];
		}
		if (p_i - p_j == 1) {
			return w.off_f[p_j];
		}
		return 0.0;
	};
	auto L = [&](int p_i, int p_j) -> double & { return w.band[p_i * (m + 1) + (p_j - p_i + m)]; };
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
	return true;
}

void f2_back_solve(const F2Grid &g, F2Work &w, const double *p_b, double *r_x) {
	const int n = g.n;
	const int m = g.m;
	auto L = [&](int p_i, int p_j) -> double { return w.band[p_i * (m + 1) + (p_j - p_i + m)]; };
	for (int i = 0; i < n; ++i) {
		const int lo = MAX(0, i - m);
		double s = p_b[i];
		for (int k = lo; k < i; ++k) {
			s -= L(i, k) * w.y[k];
		}
		w.y[i] = s / L(i, i);
	}
	for (int i = n - 1; i >= 0; --i) {
		const int hi = MIN(n, i + m + 1);
		double s = w.y[i];
		for (int k = i + 1; k < hi; ++k) {
			s -= L(k, i) * r_x[k];
		}
		r_x[i] = s / L(i, i);
	}
}

// Three right-hand sides against the one factor in one pass over it: each column's sums in f2_back_solve's own order,
// so each solution is the one f2_back_solve gives bit for bit, the band read once instead of three times
void f2_back_solve3(const F2Grid &g, F2Work &w, const double *p_b1, const double *p_b2, const double *p_b3, double *r_x1, double *r_x2, double *r_x3) {
	const int n = g.n;
	const int m = g.m;
	auto L = [&](int p_i, int p_j) -> double { return w.band[p_i * (m + 1) + (p_j - p_i + m)]; };
	double *y1 = w.y;
	double *y2 = w.y2;
	double *y3 = w.y3;
	for (int i = 0; i < n; ++i) {
		const int lo = MAX(0, i - m);
		double s1 = p_b1[i], s2 = p_b2[i], s3 = p_b3[i];
		for (int k = lo; k < i; ++k) {
			const double l = L(i, k);
			s1 -= l * y1[k];
			s2 -= l * y2[k];
			s3 -= l * y3[k];
		}
		const double d = L(i, i);
		y1[i] = s1 / d;
		y2[i] = s2 / d;
		y3[i] = s3 / d;
	}
	for (int i = n - 1; i >= 0; --i) {
		const int hi = MIN(n, i + m + 1);
		double s1 = y1[i], s2 = y2[i], s3 = y3[i];
		for (int k = i + 1; k < hi; ++k) {
			const double l = L(k, i);
			s1 -= l * r_x1[k];
			s2 -= l * r_x2[k];
			s3 -= l * r_x3[k];
		}
		const double d = L(i, i);
		r_x1[i] = s1 / d;
		r_x2[i] = s2 / d;
		r_x3[i] = s3 / d;
	}
}

// One field's contact set by complementarity (p >= 0, w >= d), at most CAP_PASSES passes from the tail's set or the
// overlap; then its tangent pressures with the last set held. As the reference, a set the cap stopped is the updated
// one while the pressure is the last solve's.
bool f2_solve_field(F2Law &r_law, F2Work &w, int p_kind, const F2Tail *p_tail) {
	const F2Grid &g = r_law.grid;
	F2Field &f = r_law.field[p_kind];
	const double alpha = p_kind == 1 ? F2_ALPHA_NS : F2_ALPHA_LUB;
	const double gam = p_kind == 1 ? F2_GAMMA_NS : F2_GAMMA_LUB;
	const bool from_tail = p_tail != nullptr && p_tail->words != nullptr && p_tail->has[p_kind] && p_tail->nx == g.nx && p_tail->ny == g.ny;
	for (int k = 0; k < g.n; ++k) {
		if (from_tail) {
			const uint32_t word = p_tail->words[p_kind * F2_TAIL_WORDS + (k >> 5)];
			f.active[k] = ((word >> (k & 31)) & 1u) != 0 && w.cont[k] && w.dom[k];
		} else {
			f.active[k] = w.cont[k] && w.dom[k] && w.d[k] > 0.0;
		}
	}
	f.settled = false;
	f.passes = 0;
	for (int it = 0; it < F2_CAP_PASSES; ++it) {
		f.passes = it + 1;
		f2_assemble_matrix(r_law, w, f.active, p_kind);
		if (!f2_factor(g, w)) {
			return false;
		}
		f2_assemble_rhs(r_law, w, f.active, p_kind == 1);
		f2_back_solve(g, w, w.rhs, w.q);
		bool changed = false;
		for (int k = 0; k < g.n; ++k) {
			const double q = w.dom[k] ? w.q[k] : 0.0;
			f.p[k] = f.active[k] ? alpha * w.Eh * w.d[k] + q : 0.0;
			const double wk = f.active[k] ? w.d[k] : -(w.h / (gam * w.E)) * q;
			bool keep = f.active[k];
			if (f.active[k] && f.p[k] < 0.0) {
				keep = false;
			}
			if (!f.active[k] && w.cont[k] && w.dom[k] && wk < w.d[k]) {
				keep = true;
			}
			w.next[k] = keep;
			changed = changed || keep != f.active[k];
		}
		if (!changed) {
			f.settled = true;
			break;
		}
		for (int k = 0; k < g.n; ++k) {
			f.active[k] = w.next[k];
		}
	}
	if (!f.settled) {
		f2_assemble_matrix(r_law, w, f.active, p_kind);
		if (!f2_factor(g, w)) {
			return false;
		}
	}
	// The field is linear in the interference for a held set: w = 1, X and Y (no held slide), combined about the centre
	for (int k = 0; k < g.n; ++k) {
		const bool a = f.active[k];
		w.rhs1[k] = a ? w.Ac[k] : 0.0;
		w.rhsx[k] = a ? r_law.X[k] * w.Ac[k] : 0.0;
		w.rhsy[k] = a ? r_law.Y[k] * w.Ac[k] : 0.0;
	}
	f2_back_solve3(g, w, w.rhs1, w.rhsx, w.rhsy, f.t1, f.tx, f.ty);
	f.any = false;
	for (int k = 0; k < g.n; ++k) {
		const bool a = f.active[k];
		f.any = f.any || a;
		f.t1[k] = a ? alpha * w.Eh + (w.dom[k] ? f.t1[k] : 0.0) : 0.0;
		f.tx[k] = a ? alpha * w.Eh * r_law.X[k] + (w.dom[k] ? f.tx[k] : 0.0) : 0.0;
		f.ty[k] = a ? alpha * w.Eh * r_law.Y[k] + (w.dom[k] ? f.ty[k] : 0.0) : 0.0;
	}
	f.solved = true;
	return true;
}

// The node's cumulative force tables, piecewise linear on the stations (stopped at the node's half chord), as the
// cell-averaged ratio dQ / dP over [e0, e1]; a cell beyond the stations takes the last interval's ratio (the bulk)
double f2_m_node(const float *p_Q, const float *p_P, double p_half, double p_e0, double p_e1) {
	double st[F2_NUM_E + 1];
	for (int i = 0; i < F2_NUM_E; ++i) {
		st[i] = MIN(F2_STATIONS[i], p_half);
	}
	st[F2_NUM_E] = p_half;
	const int last = F2_NUM_E;
	const double e0c = MIN(MAX(p_e0, 0.0), p_half);
	const double e1c = MIN(MAX(p_e1, 0.0), p_half);
	if (e1c - e0c < 1e-9 * MAX(p_half, 1.0)) {
		int j = 0;
		while (j <= last && st[j] < st[last] - 1e-12) {
			++j;
		}
		j = MAX(j - 1, 0);
		const double dq = double(p_Q[last]) - double(p_Q[j]);
		const double dp = double(p_P[last]) - double(p_P[j]);
		return dp > 1e-300 ? dq / dp : 1.0;
	}
	auto interp = [&](const float *p_f, double p_x) -> double {
		if (p_x >= st[last]) {
			return double(p_f[last]);
		}
		if (p_x <= st[0]) {
			return double(p_f[0]);
		}
		int j = 0;
		while (j + 1 < last && st[j + 1] <= p_x) {
			++j;
		}
		const double slope = (double(p_f[j + 1]) - double(p_f[j])) / (st[j + 1] - st[j]);
		return slope * (p_x - st[j]) + double(p_f[j]);
	};
	const double q1 = interp(p_Q, e1c), q0 = interp(p_Q, e0c);
	const double p1 = interp(p_P, e1c), p0 = interp(p_P, e0c);
	return (p1 - p0) > 1e-300 ? (q1 - q0) / (p1 - p0) : 1.0;
}

// The logs of the face table's log axes' nodes, made once: the same std::log of the same floats f2_log_cell took on every
// call, so every cell and coordinate comes out bit for bit as before. A function-local static is made once, by the
// first caller, even when two threads arrive together.
struct F2AxisLogs {
	double s[ftab::NUM_S];
	double c[ftab::NUM_C];
	double kg[ftab::NUM_KG];

	F2AxisLogs() {
		for (int i = 0; i < ftab::NUM_S; ++i) {
			s[i] = std::log(double(ftab::S[i]));
		}
		for (int i = 0; i < ftab::NUM_C; ++i) {
			c[i] = std::log(double(ftab::C[i]));
		}
		for (int i = 0; i < ftab::NUM_KG; ++i) {
			kg[i] = std::log(double(ftab::KG[i]));
		}
	}
};

// The node logs of a face table axis, or nullptr for an axis it does not hold
const double *f2_axis_logs(const float *p_axis, int p_n) {
	static const F2AxisLogs logs;
	if (p_axis == ftab::S && p_n == ftab::NUM_S) {
		return logs.s;
	}
	if (p_axis == ftab::C && p_n == ftab::NUM_C) {
		return logs.c;
	}
	if (p_axis == ftab::KG && p_n == ftab::NUM_KG) {
		return logs.kg;
	}
	return nullptr;
}

// The cell and coordinate of p_x on a log axis (clamped to it)
void f2_log_cell(const float *p_axis, int p_n, double p_x, int &r_i, double &r_t) {
	const double *logs = f2_axis_logs(p_axis, p_n);
	auto node_log = [&](int p_k) -> double { return logs != nullptr ? logs[p_k] : std::log(double(p_axis[p_k])); };
	const double x = MIN(MAX(p_x, double(p_axis[0])), double(p_axis[p_n - 1]));
	const double lx = std::log(x);
	int count = 0;
	for (int i = 0; i < p_n; ++i) {
		if (node_log(i) <= lx) {
			++count;
		}
	}
	const int i = MIN(MAX(count - 1, 0), p_n - 2);
	const double l0 = node_log(i), l1 = node_log(i + 1);
	r_i = i;
	r_t = MIN(MAX((lx - l0) / (l1 - l0), 0.0), 1.0);
}

// The geometry weights: plane strain (a long straight edge) wps, axisymmetric (a round contact) 1 - wps
int f2_geo_pairs(double p_wps, int r_geo[2], double r_w[2]) {
	if (p_wps >= 1.0) {
		r_geo[0] = 0;
		r_w[0] = 1.0;
		return 1;
	}
	if (p_wps <= 0.0) {
		r_geo[0] = 1;
		r_w[0] = 1.0;
		return 1;
	}
	r_geo[0] = 0;
	r_w[0] = p_wps;
	r_geo[1] = 1;
	r_w[1] = 1.0 - p_wps;
	return 2;
}

// A field's (p_kind 0 lub, 1 ns) FE multiplier at an edge type (p_edge 0: a free face, a strip or disc of half chord
// p_size; 1: an indenter edge, a strip or circular punch), cell-averaged over [e0, e1] depths from the edge
double f2_multiplier(int p_kind, int p_edge, double p_size, double p_e0, double p_e1, double p_wps) {
	const float *axis = p_edge == 0 ? ftab::S : ftab::C;
	const int na = p_edge == 0 ? ftab::NUM_S : ftab::NUM_C;
	int i;
	double t;
	f2_log_cell(axis, na, p_size, i, t);
	int geo[2];
	double wg[2];
	const int ng = f2_geo_pairs(p_wps, geo, wg);
	double out = 0.0;
	for (int gi = 0; gi < ng; ++gi) {
		const float(*Q)[ftab::NUM_Q];
		const float(*P)[ftab::NUM_Q];
		if (p_edge == 0) {
			Q = p_kind == 0 ? ftab::FREE_LUB_Q[geo[gi]] : ftab::FREE_NS_Q[geo[gi]];
			P = p_kind == 0 ? ftab::FREE_LUB_P[geo[gi]] : ftab::FREE_NS_P[geo[gi]];
		} else {
			Q = p_kind == 0 ? ftab::IND_LUB_Q[geo[gi]] : ftab::IND_NS_Q[geo[gi]];
			P = p_kind == 0 ? ftab::IND_LUB_P[geo[gi]] : ftab::IND_NS_P[geo[gi]];
		}
		const double m0 = f2_m_node(Q[i], P[i], double(axis[i]), p_e0, p_e1);
		const double m1 = f2_m_node(Q[i + 1], P[i + 1], double(axis[i + 1]), p_e0, p_e1);
		out += wg[gi] * ((1.0 - t) * m0 + t * m1);
	}
	return out;
}

// The face axis's cell (mu 0, 0.3, 0.6, 0.9 and the bonded column)
void f2_mu_cell(double p_mu, int &r_f, double &r_t) {
	const int n = ftab::NUM_MU;
	if (p_mu >= double(ftab::MU[n - 1])) {
		r_f = n - 2;
		r_t = 1.0;
		return;
	}
	const double mu = MAX(p_mu, 0.0);
	int count = 0;
	for (int i = 0; i < n; ++i) {
		if (double(ftab::MU[i]) <= mu) {
			++count;
		}
	}
	const int f = MIN(MAX(count - 1, 0), n - 2);
	r_f = f;
	r_t = (mu - double(ftab::MU[f])) / (double(ftab::MU[f + 1]) - double(ftab::MU[f]));
}

// The FE's friction reshaping at an edge type for a Coulomb face: identity at mu 0 and at the bonded column, the FE's own
// profile at 0.3, 0.6 and 0.9 against the blended field, linear in mu between
double f2_mix_multiplier(int p_edge, double p_size, double p_mu, double p_e0, double p_e1, double p_wps) {
	const int nmu = ftab::NUM_MU;
	if (p_mu <= 0.0 || p_mu >= double(ftab::MU[nmu - 1])) {
		return 1.0;
	}
	int f;
	double tf;
	f2_mu_cell(p_mu, f, tf);
	const float *axis = p_edge == 0 ? ftab::S : ftab::C;
	const int na = p_edge == 0 ? ftab::NUM_S : ftab::NUM_C;
	int i;
	double t;
	f2_log_cell(axis, na, p_size, i, t);
	int geo[2];
	double wg[2];
	const int ng = f2_geo_pairs(p_wps, geo, wg);
	double out = 0.0;
	for (int gi = 0; gi < ng; ++gi) {
		auto col = [&](int p_m) -> double {
			if (p_m == 0 || p_m == nmu - 1) {
				return 1.0;
			}
			const float *Q0 = p_edge == 0 ? ftab::FREE_MIX_Q[geo[gi]][p_m - 1][i] : ftab::IND_MIX_Q[geo[gi]][p_m - 1][i];
			const float *P0 = p_edge == 0 ? ftab::FREE_MIX_P[geo[gi]][p_m - 1][i] : ftab::IND_MIX_P[geo[gi]][p_m - 1][i];
			const float *Q1 = p_edge == 0 ? ftab::FREE_MIX_Q[geo[gi]][p_m - 1][i + 1] : ftab::IND_MIX_Q[geo[gi]][p_m - 1][i + 1];
			const float *P1 = p_edge == 0 ? ftab::FREE_MIX_P[geo[gi]][p_m - 1][i + 1] : ftab::IND_MIX_P[geo[gi]][p_m - 1][i + 1];
			const double a = f2_m_node(Q0, P0, double(axis[i]), p_e0, p_e1);
			const double b = f2_m_node(Q1, P1, double(axis[i + 1]), p_e0, p_e1);
			return (1.0 - t) * a + t * b;
		};
		out += wg[gi] * ((1.0 - tf) * col(f) + tf * col(f + 1));
	}
	return out;
}

// The FE's friction share at an edge type: p_edge 0 a free face (S, mu, K/G), 1 an indenter edge (c, mu), 2 a smooth edge
// (mu and tau = |grad d| h / d_max, linear in tau between the two calibrated wedges, clamped)
double f2_phi(int p_edge, double p_size, double p_mu, double p_kg, double p_wps, double p_tau) {
	int f;
	double tf;
	f2_mu_cell(p_mu, f, tf);
	if (p_edge == 2) {
		const double t0 = double(ftab::PHI_SMOOTH_TAU[0]), t1 = double(ftab::PHI_SMOOTH_TAU[1]);
		const double u = MIN(MAX((p_tau - t0) / (t1 - t0), 0.0), 1.0);
		const double rf = (1.0 - u) * double(ftab::PHI_SMOOTH[0][f]) + u * double(ftab::PHI_SMOOTH[1][f]);
		const double rf1 = (1.0 - u) * double(ftab::PHI_SMOOTH[0][f + 1]) + u * double(ftab::PHI_SMOOTH[1][f + 1]);
		return (1.0 - tf) * rf + tf * rf1;
	}
	const float *axis = p_edge == 0 ? ftab::S : ftab::C;
	const int na = p_edge == 0 ? ftab::NUM_S : ftab::NUM_C;
	int i;
	double t;
	f2_log_cell(axis, na, p_size, i, t);
	int geo[2];
	double wg[2];
	const int ng = f2_geo_pairs(p_wps, geo, wg);
	double r = 0.0;
	for (int gi = 0; gi < ng; ++gi) {
		if (p_edge == 0) {
			int kc;
			double tk;
			f2_log_cell(ftab::KG, ftab::NUM_KG, p_kg, kc, tk);
			const int kk[2] = { kc, kc + 1 };
			const double wk[2] = { 1.0 - tk, tk };
			const int ff[2] = { f, f + 1 };
			const double wf[2] = { 1.0 - tf, tf };
			const int ii[2] = { i, i + 1 };
			const double wi[2] = { 1.0 - t, t };
			for (int a = 0; a < 2; ++a) {
				for (int b = 0; b < 2; ++b) {
					for (int c = 0; c < 2; ++c) {
						r += wg[gi] * wk[a] * wf[b] * wi[c] * double(ftab::PHI_FREE[geo[gi]][kk[a]][ff[b]][ii[c]]);
					}
				}
			}
		} else {
			const int ff[2] = { f, f + 1 };
			const double wf[2] = { 1.0 - tf, tf };
			const int ii[2] = { i, i + 1 };
			const double wi[2] = { 1.0 - t, t };
			for (int b = 0; b < 2; ++b) {
				for (int c = 0; c < 2; ++c) {
					r += wg[gi] * wf[b] * wi[c] * double(ftab::PHI_IND[geo[gi]][ff[b]][ii[c]]);
				}
			}
		}
	}
	return r;
}

// The cell and coordinate of p_x on a linear axis (clamped to it)
void f3_lin_cell(const float *p_axis, int p_n, double p_x, int &r_j, double &r_u) {
	const double x = MIN(MAX(p_x, double(p_axis[0])), double(p_axis[p_n - 1]));
	int count = 0;
	for (int i = 0; i < p_n; ++i) {
		if (double(p_axis[i]) <= x) {
			++count;
		}
	}
	const int j = MIN(MAX(count - 1, 0), p_n - 2);
	r_j = j;
	r_u = MIN(MAX((x - double(p_axis[j])) / (double(p_axis[j + 1]) - double(p_axis[j])), 0.0), 1.0);
}

// An indenter edge's cell-averaged multiplier at margin p_m (depths): the inset tables, linear in ln(size) on the S nodes
// and linear in the margin, the geometries weighted as f2_multiplier
double f3_inset_multiplier(int p_kind, double p_size, double p_m, double p_e0, double p_e1, double p_wps) {
	int j;
	double u;
	f3_lin_cell(ftab::INSET_M, ftab::NUM_M, p_m, j, u);
	int geo[2];
	double wg[2];
	const int ng = f2_geo_pairs(p_wps, geo, wg);
	double out = 0.0;
	for (int gi = 0; gi < ng; ++gi) {
		int i;
		double t;
		f2_log_cell(ftab::S, ftab::NUM_S, p_size, i, t);
		const float(*Q)[ftab::NUM_M][ftab::NUM_Q] = p_kind == 0 ? ftab::INSET_LUB_Q[geo[gi]] : ftab::INSET_NS_Q[geo[gi]];
		const float(*P)[ftab::NUM_M][ftab::NUM_Q] = p_kind == 0 ? ftab::INSET_LUB_P[geo[gi]] : ftab::INSET_NS_P[geo[gi]];
		const int ii[2] = { i, i + 1 };
		const double wi[2] = { 1.0 - t, t };
		const int jj[2] = { j, j + 1 };
		const double wj[2] = { 1.0 - u, u };
		double v = 0.0;
		for (int a = 0; a < 2; ++a) {
			for (int b = 0; b < 2; ++b) {
				if (wi[a] * wj[b] != 0.0) {
					v += wi[a] * wj[b] * f2_m_node(Q[ii[a]][jj[b]], P[ii[a]][jj[b]], double(ftab::S[ii[a]]), p_e0, p_e1);
				}
			}
		}
		out += wg[gi] * v;
	}
	return out;
}

// An indenter edge's friction share at margin p_m (the FE's), interpolated as f2_phi with the margin axis
double f3_inset_phi(double p_size, double p_m, double p_mu, double p_kg, double p_wps) {
	int f;
	double tf;
	f2_mu_cell(p_mu, f, tf);
	int j;
	double u;
	f3_lin_cell(ftab::INSET_M, ftab::NUM_M, p_m, j, u);
	int kc;
	double tk;
	f2_log_cell(ftab::KG, ftab::NUM_KG, p_kg, kc, tk);
	int geo[2];
	double wg[2];
	const int ng = f2_geo_pairs(p_wps, geo, wg);
	double r = 0.0;
	for (int gi = 0; gi < ng; ++gi) {
		int i;
		double t;
		f2_log_cell(ftab::S, ftab::NUM_S, p_size, i, t);
		const int kk[2] = { kc, kc + 1 };
		const double wk[2] = { 1.0 - tk, tk };
		const int ff[2] = { f, f + 1 };
		const double wf[2] = { 1.0 - tf, tf };
		const int ii[2] = { i, i + 1 };
		const double wi[2] = { 1.0 - t, t };
		const int jj[2] = { j, j + 1 };
		const double wj[2] = { 1.0 - u, u };
		for (int a = 0; a < 2; ++a) {
			for (int b = 0; b < 2; ++b) {
				for (int c = 0; c < 2; ++c) {
					for (int d = 0; d < 2; ++d) {
						r += wg[gi] * wk[a] * wf[b] * wi[c] * wj[d] * double(ftab::PHI_INSET[geo[gi]][kk[a]][ff[b]][ii[c]][jj[d]]);
					}
				}
			}
		}
	}
	return r;
}

// The area of the convex counter-clockwise p_poly clipped to the rectangle (the polygon clipped by the rectangle's four
// edges: a clipped polygon can carry an edge of no length, which as a clipping edge would keep nothing)
double f3_rect_area(const Poly &p_poly, double p_x0, double p_x1, double p_y0, double p_y1) {
	Poly cut;
	clip_convex(p_poly, rect(p_x0, p_x1, p_y0, p_y1), cut);
	if (cut.n < 3) {
		return 0.0;
	}
	return std::abs(signed_area(cut));
}

// A convex counter-clockwise polygon's edges as f3_exit reads them: each edge's unit outward normal and its offset along
// it, made once a polygon (the same expressions f3_exit took for every edge on every call); ok false for an edge of no
// length, which f3_exit skips
struct F3Edges {
	int n = 0;
	double nx[MAX_POLY];
	double ny[MAX_POLY];
	double c[MAX_POLY];
	bool ok[MAX_POLY];
};

void f3_edges(const Poly &p_ccw, F3Edges &r_edges) {
	r_edges.n = p_ccw.n;
	for (int k = 0; k < p_ccw.n; ++k) {
		const int k1 = (k + 1) % p_ccw.n;
		const double ex = p_ccw.x[k1] - p_ccw.x[k], ey = p_ccw.y[k1] - p_ccw.y[k];
		const double L = std::sqrt(ex * ex + ey * ey);
		r_edges.ok[k] = L > 0.0;
		r_edges.nx[k] = r_edges.ok[k] ? ey / L : 0.0;
		r_edges.ny[k] = r_edges.ok[k] ? -ex / L : 0.0;
		r_edges.c[k] = r_edges.ok[k] ? r_edges.nx[k] * p_ccw.x[k] + r_edges.ny[k] * p_ccw.y[k] : 0.0;
	}
}

// The distance from an inside point along the unit axis direction (p_dx, p_dy) to the convex counter-clockwise polygon's edge
double f3_exit(const F3Edges &p_edges, double p_px, double p_py, double p_dx, double p_dy) {
	double t = std::numeric_limits<double>::infinity();
	for (int k = 0; k < p_edges.n; ++k) {
		if (!p_edges.ok[k]) {
			continue;
		}
		const double nd = p_edges.nx[k] * p_dx + p_edges.ny[k] * p_dy;
		if (nd > 1e-15) {
			t = MIN(t, (p_edges.c[k] - (p_edges.nx[k] * p_px + p_edges.ny[k] * p_py)) / nd);
		}
	}
	return MAX(t, 0.0);
}

// The same through the union of coplanar partner faces (p_edges: each face's edges): from the face holding the point,
// then on through any face that holds the point it leaves by
double f3_exit_union(const PartnerSet &p_set, const F3Edges *p_edges, double p_px, double p_py, double p_dx, double p_dy, double p_eps) {
	double tot = 0.0;
	double x = p_px, y = p_py;
	bool used[MAX_PARTNERS] = {};
	for (int it = 0; it < p_set.n; ++it) {
		int j = -1;
		for (int i = 0; i < p_set.n && j < 0; ++i) {
			if (!used[i] && inside_convex(p_set.face[i], it == 0 ? x : x + p_eps * p_dx, it == 0 ? y : y + p_eps * p_dy)) {
				j = i;
			}
		}
		if (j < 0) {
			break;
		}
		used[j] = true;
		const double t = f3_exit(p_edges[j], x, y, p_dx, p_dy);
		tot += t;
		x += t * p_dx;
		y += t * p_dy;
		if (p_set.n == 1) {
			break;
		}
	}
	return tot;
}

// The interval list of an edge stretch, less the parts another coplanar partner face covers just outside it (an edge two
// partner faces share is inside their union, not an indenter edge)
int f2_outside_others(const PartnerSet &p_partners, int p_self, double p_ax, double p_ay, double p_bx, double p_by, double p_t0, double p_t1, double p_eps, double r_t[][2]) {
	int n = 1;
	r_t[0][0] = p_t0;
	r_t[0][1] = p_t1;
	const double ex = p_bx - p_ax, ey = p_by - p_ay;
	const double len = std::sqrt(ex * ex + ey * ey);
	if (!(len > 0.0)) {
		return 0;
	}
	// a counter-clockwise face's outward normal is (ey, -ex) / len
	const double ox = p_eps * ey / len, oy = -p_eps * ex / len;
	for (int j = 0; j < p_partners.n && n > 0; ++j) {
		if (j == p_self) {
			continue;
		}
		double u0, u1;
		if (!f2_clip_segment(p_ax + ox, p_ay + oy, p_bx + ox, p_by + oy, p_partners.face[j], 0.0, u0, u1)) {
			continue;
		}
		double out[2 * MAX_PARTNERS + 2][2];
		int m = 0;
		for (int s = 0; s < n; ++s) {
			if (u1 <= r_t[s][0] || u0 >= r_t[s][1]) {
				out[m][0] = r_t[s][0];
				out[m][1] = r_t[s][1];
				++m;
				continue;
			}
			if (u0 > r_t[s][0] + 1e-9) {
				out[m][0] = r_t[s][0];
				out[m][1] = u0;
				++m;
			}
			if (u1 < r_t[s][1] - 1e-9) {
				out[m][0] = u1;
				out[m][1] = r_t[s][1];
				++m;
			}
		}
		n = MIN(m, 2 * MAX_PARTNERS + 2);
		for (int s = 0; s < n; ++s) {
			r_t[s][0] = out[s][0];
			r_t[s][1] = out[s][1];
		}
	}
	return n;
}

// Every cell's distance to the nearest stretch a -> b of an edge type, its normal's |nx| and |ny| and the cell's extent
// along that normal (a strictly nearer stretch replaces a nearer-or-equal one, in edge order)
void f2_edge_stretch(const F2Law &p_law, F2Work &w, int p_type, double p_ax, double p_ay, double p_bx, double p_by) {
	const double vx = p_bx - p_ax, vy = p_by - p_ay;
	const double L2 = vx * vx + vy * vy;
	const double Ln = std::hypot(vx, vy);
	const double nxv = Ln > 0.0 ? vy / Ln : 1.0;
	const double nyv = Ln > 0.0 ? -vx / Ln : 0.0;
	for (int k = 0; k < p_law.grid.n; ++k) {
		double t = ((p_law.X[k] - p_ax) * vx + (p_law.Y[k] - p_ay) * vy) / MAX(L2, 1e-300);
		t = MIN(MAX(t, 0.0), 1.0);
		const double dx = p_law.X[k] - (p_ax + t * vx);
		const double dy = p_law.Y[k] - (p_ay + t * vy);
		const double dist = std::hypot(dx, dy);
		if (dist < w.e[p_type][k]) {
			w.e[p_type][k] = dist;
			w.ext[p_type][k] = std::abs(nxv) * p_law.DX[k] + std::abs(nyv) * p_law.DY[k];
			w.anx[p_type][k] = std::abs(nxv);
			w.any[p_type][k] = std::abs(nyv);
			if (p_type == 1) {
				w.px[k] = p_ax + t * vx;
				w.py[k] = p_ay + t * vy;
			}
		}
	}
}

// The face law v3 for one patch: the soft face p_soft, the partner's coplanar faces p_partners (counter-clockwise), the
// overlap of the undeformed surfaces d = d0 + gx x + gy y, the layer (E, K, h), the pair's static mu. p_edge_tol in
// depths (F2_EDGE_TOL). p_tail: the last step's contact sets and the held slide's memory, or null. False when the patch
// presses nothing.
bool f2_face_law(const Poly &p_soft, const PartnerSet &p_partners, double p_d0, double p_gx, double p_gy, double p_E, double p_K, double p_h, double p_mu, double p_edge_tol, const F2Tail *p_tail, F2Law &r_law) {
	thread_local F2Work scratch_work;
	F2Work &w = scratch_work;
	r_law.empty = true;
	r_law.field[0].solved = false;
	r_law.field[1].solved = false;
	if (!(p_E > 0.0) || !(p_h > 0.0) || !(9.0 * p_K > p_E) || p_partners.n < 1) {
		return false;
	}
	const double h = p_h;
	w.E = p_E;
	w.h = h;
	w.G = 3.0 * p_K * p_E / (9.0 * p_K - p_E);
	const double nu = (3.0 * p_K - 2.0 * w.G) / (2.0 * (3.0 * p_K + w.G));
	const double kW = p_E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * h);
	w.Eh = p_E / h;
	w.kq = kW - p_E / h;
	w.kg = p_K / w.G;
	if (!(w.kq > 0.0)) {
		return false;
	}

	// The contactable region: the soft face clipped by each partner face
	Poly soft = p_soft;
	make_ccw(soft);
	thread_local PartnerSet scratch_partners;
	thread_local Poly scratch_pieces[MAX_PARTNERS];
	PartnerSet &partners = scratch_partners;
	partners = p_partners;
	Poly *pieces = scratch_pieces;
	int num_pieces = 0;
	auto clip_pieces = [&]() {
		num_pieces = 0;
		for (int i = 0; i < partners.n; ++i) {
			make_ccw(partners.face[i]);
			clip_convex(soft, partners.face[i], pieces[num_pieces]);
			if (pieces[num_pieces].n >= 3) {
				++num_pieces;
			}
		}
	};
	clip_pieces();
	if (num_pieces == 0) {
		return false;
	}

	// The held slide's memory (rule 6): the gradient the contact set grew at, in the caller's frame
	const bool has_held = p_tail != nullptr && p_tail->has_held && p_tail->held[2] > 0.0;
	double hgx = has_held ? p_tail->held[0] : 0.0, hgy = has_held ? p_tail->held[1] : 0.0;

	// The grid's turn (rule 3): the circular mean of 4 alpha over the contactable region's edges (an edge another piece
	// covers is inside the region) weighted L exp(-(Lmax - L) / (EPS Lmax)), when its length is at least AGREE of the
	// weights' sum; else the first longest edge. Reduced to [-45, 45) deg; the figures are turned back at the end.
	double gx = p_gx, gy = p_gy;
	{
		double best = -1.0, best_ang = 0.0;
		auto each_edge = [&](auto &&p_fn) {
			for (int i = 0; i < num_pieces; ++i) {
				const Poly &pc = pieces[i];
				for (int k = 0; k < pc.n; ++k) {
					const int k1 = (k + 1) % pc.n;
					const double ex = pc.x[k1] - pc.x[k], ey = pc.y[k1] - pc.y[k];
					const double len = std::hypot(ex, ey);
					if (!(len > 0.0)) {
						continue;
					}
					if (num_pieces > 1) {
						const double mx = 0.5 * (pc.x[k] + pc.x[k1]) + 1e-6 * h * ey / len;
						const double my = 0.5 * (pc.y[k] + pc.y[k1]) - 1e-6 * h * ex / len;
						bool inner = false;
						for (int j = 0; j < num_pieces && !inner; ++j) {
							inner = j != i && inside_convex(pieces[j], mx, my);
						}
						if (inner) {
							continue;
						}
					}
					p_fn(len, std::atan2(ey, ex));
				}
			}
		};
		each_edge([&](double p_len, double p_ang) {
			if (p_len > best) {
				best = p_len;
				best_ang = p_ang;
			}
		});
		double zr = 0.0, zi = 0.0, ws = 0.0;
		each_edge([&](double p_len, double p_ang) {
			const double wgt = p_len * std::exp(-(best - p_len) / (F3_ALIGN_EPS * best));
			zr += wgt * std::cos(4.0 * p_ang);
			zi += wgt * std::sin(4.0 * p_ang);
			ws += wgt;
		});
		const double ang = std::hypot(zr, zi) >= F3_ALIGN_AGREE * ws ? 0.25 * std::atan2(zi, zr) : best_ang;
		r_law.rot = ang - (PI / 2.0) * std::nearbyint(ang / (PI / 2.0));
		if (std::abs(r_law.rot) > 1e-12) {
			const double cr = std::cos(-r_law.rot), sn = std::sin(-r_law.rot);
			auto turn = [&](Poly &r_p) {
				for (int k = 0; k < r_p.n; ++k) {
					const double x = r_p.x[k], y = r_p.y[k];
					r_p.x[k] = cr * x - sn * y;
					r_p.y[k] = sn * x + cr * y;
				}
			};
			turn(soft);
			for (int i = 0; i < partners.n; ++i) {
				turn(partners.face[i]);
			}
			const double gx0 = gx;
			gx = cr * gx0 - sn * gy;
			gy = sn * gx0 + cr * gy;
			const double hg0 = hgx;
			hgx = cr * hg0 - sn * hgy;
			hgy = sn * hg0 + cr * hgy;
			clip_pieces();
			if (num_pieces == 0) {
				return false;
			}
		} else {
			r_law.rot = 0.0;
		}
	}

	// The engine grid: inner lines over the contactable region's bounding box, outer lines to the soft face or BOX depths
	double ox0 = 0.0, ox1 = 0.0, oy0 = 0.0, oy1 = 0.0;
	PolyMoments om;
	double dmax = -1e300, hmax = -1e300;
	for (int i = 0; i < num_pieces; ++i) {
		double a0, a1, b0, b1;
		poly_bounds(pieces[i], a0, a1, b0, b1);
		ox0 = i == 0 ? a0 : MIN(ox0, a0);
		ox1 = i == 0 ? a1 : MAX(ox1, a1);
		oy0 = i == 0 ? b0 : MIN(oy0, b0);
		oy1 = i == 0 ? b1 : MAX(oy1, b1);
		const PolyMoments mi = poly_moments(pieces[i]);
		om.A += mi.A;
		om.Sx += mi.Sx;
		om.Sy += mi.Sy;
		om.Sxx += mi.Sxx;
		om.Syy += mi.Syy;
		for (int k = 0; k < pieces[i].n; ++k) {
			dmax = MAX(dmax, p_d0 + gx * pieces[i].x[k] + gy * pieces[i].y[k]);
			hmax = MAX(hmax, hgx * pieces[i].x[k] + hgy * pieces[i].y[k]);
		}
	}
	double sx0, sx1, sy0, sy1;
	poly_bounds(soft, sx0, sx1, sy0, sy1);
	// A side of the contactable region within the edge tolerance of the soft face's is the soft face's: a partner the size
	// of its pad, its edges a float's noise inside the pad's, leaves no sliver of free layer beside them
	const double tol = p_edge_tol * h;
	if (tol > 0.0) {
		ox0 = ox0 - sx0 <= tol ? sx0 : ox0;
		ox1 = sx1 - ox1 <= tol ? sx1 : ox1;
		oy0 = oy0 - sy0 <= tol ? sy0 : oy0;
		oy1 = sy1 - oy1 <= tol ? sy1 : oy1;
	}
	const double box = F2_BOX * h;
	const double bx0 = MAX(ox0 - box, sx0), bx1 = MIN(ox1 + box, sx1), by0 = MAX(oy0 - box, sy0), by1 = MIN(oy1 + box, sy1);
	F2Grid &g = r_law.grid;
	int nxl = f2_engine_lines(bx0, ox0, ox1, bx1, h, 0, g.xe);
	int nyl = f2_engine_lines(by0, oy0, oy1, by1, h, 0, g.ye);
	if (nxl < 2 || nyl < 2) {
		return false;
	}
	// The cell cap. An axis's lines never hold more inner cells than the even count they were asked for (f2_inner_lines),
	// so each pass that asks for two fewer on the larger axis shrinks the grid, and the passes end by N_MIN
	while ((nxl - 1) * (nyl - 1) > F2_MAX_CELLS) {
		int nx_i = 0, ny_i = 0;
		for (int k = 0; k + 1 < nxl; ++k) {
			nx_i += (g.xe[k] >= ox0 - 1e-9 && g.xe[k + 1] <= ox1 + 1e-9) ? 1 : 0;
		}
		for (int k = 0; k + 1 < nyl; ++k) {
			ny_i += (g.ye[k] >= oy0 - 1e-9 && g.ye[k + 1] <= oy1 + 1e-9) ? 1 : 0;
		}
		if (MAX(nx_i, ny_i) <= F2_N_MIN) {
			break;
		}
		if (nx_i >= ny_i) {
			nxl = f2_engine_lines(bx0, ox0, ox1, bx1, h, nx_i - 2, g.xe);
		} else {
			nyl = f2_engine_lines(by0, oy0, oy1, by1, h, ny_i - 2, g.ye);
		}
		if (nxl < 2 || nyl < 2) {
			return false;
		}
	}
	g.nx = nxl - 1;
	g.ny = nyl - 1;
	g.n = g.nx * g.ny;
	if (g.n > F2_MAX_CELLS) {
		return false;
	}
	g.y_fast = g.ny <= g.nx;
	g.m = g.y_fast ? g.ny : g.nx;
	if (g.m + 1 > F2_MAX_BAND) {
		return false;
	}

	// The cells: centres, sizes, the layer's own face, the partner's, the interference and the held slide
	const double gn = std::hypot(gx, gy);
	const double hn = std::hypot(hgx, hgy);
	for (int i = 0; i < g.nx; ++i) {
		for (int j = 0; j < g.ny; ++j) {
			const int k = g.k(i, j);
			r_law.X[k] = 0.5 * (g.xe[i + 1] + g.xe[i]);
			r_law.Y[k] = 0.5 * (g.ye[j + 1] + g.ye[j]);
			r_law.DX[k] = g.xe[i + 1] - g.xe[i];
			r_law.DY[k] = g.ye[j + 1] - g.ye[j];
			w.area[k] = r_law.DX[k] * r_law.DY[k];
			w.dom[k] = inside_convex(soft, r_law.X[k], r_law.Y[k]);
			w.cont[k] = w.dom[k] && inside_any(partners, r_law.X[k], r_law.Y[k]);
			w.d[k] = p_d0 + gx * r_law.X[k] + gy * r_law.Y[k];
			w.hx[k] = 0.0;
			w.hy[k] = 0.0;
			if (has_held) {
				// the slide each point made while free, by the remembered gradient
				if (hn > 1e-15) {
					const double s = MAX(hmax - (hgx * r_law.X[k] + hgy * r_law.Y[k]), 0.0) / hn;
					const double mag = F2_KS * hn * s * std::pow(s / h, F2_KP - 1.0);
					w.hx[k] = -mag * hgx / hn;
					w.hy[k] = -mag * hgy / hn;
				}
			} else if (gn > 1e-15) {
				const double s = MAX(dmax - w.d[k], 0.0) / gn;
				const double mag = F2_KS * gn * s * std::pow(s / h, F2_KP - 1.0);
				w.hx[k] = -mag * gx / gn;
				w.hy[k] = -mag * gy / gn;
			}
			for (int t = 0; t < 2; ++t) {
				w.e[t][k] = std::numeric_limits<double>::infinity();
				w.ext[t][k] = 0.0;
				w.anx[t][k] = 1.0;
				w.any[t][k] = 0.0;
			}
			w.px[k] = 0.0;
			w.py[k] = 0.0;
		}
	}

	// The edges: the layer's own edges where the partner presses them (free faces), the partner's edges inside the
	// layer's face (indenter edges), each stretch by Cyrus and Beck
	for (int k = 0; k < soft.n; ++k) {
		const int k1 = (k + 1) % soft.n;
		for (int j = 0; j < partners.n; ++j) {
			double t0, t1;
			if (!f2_clip_segment(soft.x[k], soft.y[k], soft.x[k1], soft.y[k1], partners.face[j], tol, t0, t1)) {
				continue;
			}
			const double ex = soft.x[k1] - soft.x[k], ey = soft.y[k1] - soft.y[k];
			f2_edge_stretch(r_law, w, 0, soft.x[k] + t0 * ex, soft.y[k] + t0 * ey, soft.x[k] + t1 * ex, soft.y[k] + t1 * ey);
		}
	}
	for (int j = 0; j < partners.n; ++j) {
		const Poly &pf = partners.face[j];
		for (int k = 0; k < pf.n; ++k) {
			const int k1 = (k + 1) % pf.n;
			double t0, t1;
			if (!f2_clip_segment(pf.x[k], pf.y[k], pf.x[k1], pf.y[k1], soft, -tol, t0, t1)) {
				continue;
			}
			double parts[2 * MAX_PARTNERS + 2][2];
			const int np = partners.n > 1 ? f2_outside_others(partners, j, pf.x[k], pf.y[k], pf.x[k1], pf.y[k1], t0, t1, MAX(1e-6 * h, tol), parts) : 1;
			if (partners.n == 1) {
				parts[0][0] = t0;
				parts[0][1] = t1;
			}
			const double ex = pf.x[k1] - pf.x[k], ey = pf.y[k1] - pf.y[k];
			for (int s = 0; s < np; ++s) {
				if (parts[s][1] - parts[s][0] > 1e-9) {
					f2_edge_stretch(r_law, w, 1, pf.x[k] + parts[s][0] * ex, pf.y[k] + parts[s][0] * ey, pf.x[k] + parts[s][1] * ex, pf.y[k] + parts[s][1] * ey);
				}
			}
		}
	}
	// Each cell's nearest indenter edge's margin (rule 1): from that edge's nearest point to the layer's own edge
	const double lam = F3_INSET_LAMBDA;
	for (int k = 0; k < g.n; ++k) {
		double m = std::numeric_limits<double>::infinity();
		if (std::isfinite(w.e[1][k])) {
			for (int s = 0; s < soft.n; ++s) {
				const int s1 = (s + 1) % soft.n;
				const double vx = soft.x[s1] - soft.x[s], vy = soft.y[s1] - soft.y[s];
				double t = ((w.px[k] - soft.x[s]) * vx + (w.py[k] - soft.y[s]) * vy) / MAX(vx * vx + vy * vy, 1e-300);
				t = MIN(MAX(t, 0.0), 1.0);
				m = MIN(m, std::hypot(w.px[k] - (soft.x[s] + t * vx), w.py[k] - (soft.y[s] + t * vy)));
			}
		}
		w.marg[k] = MIN(m / h, F2_BOX);
		w.chi[k] = 1.0 - std::exp(-MIN(MAX(w.marg[k], 0.0), F2_BOX) / lam);
		w.stair_ind[k] = (w.anx[1][k] + w.any[1][k]) * w.chi[k];
	}

	// The embedded edges (rule 4): the areas each cell stands for, and where each axis of a cell meets the edges
	for (int i = 0; i < g.nx; ++i) {
		for (int j = 0; j < g.ny; ++j) {
			const int k = g.k(i, j);
			const double x0 = g.xe[i], x1 = g.xe[i + 1], y0 = g.ye[j], y1 = g.ye[j + 1];
			const double fd = MIN(MAX(f3_rect_area(soft, x0, x1, y0, y1) / w.area[k], 0.0), 1.0);
			double fca = 0.0;
			for (int p = 0; p < num_pieces; ++p) {
				fca += f3_rect_area(pieces[p], x0, x1, y0, y1);
			}
			const double fc = MIN(MAX(fca / w.area[k], 0.0), 1.0);
			// kept in the arrays the matrix reads: the cell's own covered parts first, the shares added below
			w.Ad[k] = w.dom[k] ? fd * w.area[k] : 0.0;
			w.Ac[k] = w.cont[k] ? fc * w.area[k] : 0.0;
			w.recv_d[k] = false;
			w.recv_c[k] = false;
			w.rhs1[k] = fd; // scratch: the covered fractions, read by the share pass
			w.rhsx[k] = fc;
		}
	}
	for (int i = 0; i < g.nx; ++i) {
		for (int j = 0; j < g.ny; ++j) {
			const int k = g.k(i, j);
			const double fd = w.rhs1[k], fc = w.rhsx[k];
			int nb[4];
			int nn = 0;
			const int di[4] = { -1, 1, 0, 0 }, dj[4] = { 0, 0, -1, 1 };
			for (int s = 0; s < 4; ++s) {
				const int ni = i + di[s], nj = j + dj[s];
				if (ni >= 0 && ni < g.nx && nj >= 0 && nj < g.ny) {
					nb[nn++] = g.k(ni, nj);
				}
			}
			if (!w.dom[k] && fd > 0.0) {
				// a cell whose centre is outside the face gives its covered part to the neighbours that keep theirs
				int cnt = 0;
				for (int s = 0; s < nn; ++s) {
					cnt += w.dom[nb[s]] ? 1 : 0;
				}
				if (cnt > 0) {
					const double sd = fd * w.area[k] / cnt, sc = fc * w.area[k] / cnt;
					for (int s = 0; s < nn; ++s) {
						const int q = nb[s];
						if (!w.dom[q]) {
							continue;
						}
						w.Ad[q] += sd;
						w.recv_d[q] = true;
						if (w.cont[q]) {
							w.Ac[q] += sc;
							w.recv_c[q] = true;
						}
					}
				}
			} else if (w.dom[k] && !w.cont[k] && fc > 0.0) {
				// a cell in the face but not under the partner gives its partner-covered part to the neighbours under it
				int cnt = 0;
				for (int s = 0; s < nn; ++s) {
					cnt += w.cont[nb[s]] ? 1 : 0;
				}
				if (cnt > 0) {
					const double sc = fc * w.area[k] / cnt;
					w.Ad[k] -= fc * w.area[k];
					for (int s = 0; s < nn; ++s) {
						const int q = nb[s];
						if (!w.cont[q]) {
							continue;
						}
						w.Ad[q] += sc;
						w.Ac[q] += sc;
						w.recv_c[q] = true;
					}
				}
			}
		}
	}
	const double dir_x[4] = { -1.0, 1.0, 0.0, 0.0 }, dir_y[4] = { 0.0, 0.0, -1.0, 1.0 };
	// each polygon's edges once, not once a cell and direction
	thread_local F3Edges scratch_soft_edges;
	thread_local F3Edges scratch_partner_edges[MAX_PARTNERS];
	F3Edges &soft_edges = scratch_soft_edges;
	F3Edges *partner_edges = scratch_partner_edges;
	f3_edges(soft, soft_edges);
	for (int j = 0; j < partners.n; ++j) {
		f3_edges(partners.face[j], partner_edges[j]);
	}
	for (int k = 0; k < g.n; ++k) {
		for (int s = 0; s < 4; ++s) {
			w.dS[k][s] = w.dom[k] ? f3_exit(soft_edges, r_law.X[k], r_law.Y[k], dir_x[s], dir_y[s]) : 0.0;
			w.dP[k][s] = w.cont[k] ? f3_exit_union(partners, partner_edges, r_law.X[k], r_law.Y[k], dir_x[s], dir_y[s], 1e-9 * h) : 0.0;
		}
	}

	// The no-slip field first: its contact set gives the contact's size and the friction share
	F2Field &ns = r_law.field[1];
	F2Field &lub = r_law.field[0];
	if (!f2_solve_field(r_law, w, 1, p_tail)) {
		return false;
	}
	double Fref = 0.0;
	double A_act = 0.0, sxa = 0.0, sya = 0.0;
	for (int k = 0; k < g.n; ++k) {
		Fref += ns.p[k] * w.Ac[k];
		if (ns.active[k]) {
			A_act += w.Ac[k];
			sxa += w.Ac[k] * r_law.X[k];
			sya += w.Ac[k] * r_law.Y[k];
		}
	}
	if (!ns.any || !(Fref > 0.0) || !(A_act > 0.0)) {
		return false;
	}
	const double cxa = sxa / A_act, cya = sya / A_act;
	double Iyy_a = 0.0, Ixx_a = 0.0;
	for (int k = 0; k < g.n; ++k) {
		if (ns.active[k]) {
			Iyy_a += w.Ac[k] * ((r_law.X[k] - cxa) * (r_law.X[k] - cxa) + r_law.DX[k] * r_law.DX[k] / 12.0);
			Ixx_a += w.Ac[k] * ((r_law.Y[k] - cya) * (r_law.Y[k] - cya) + r_law.DY[k] * r_law.DY[k] / 12.0);
		}
	}
	auto half_dims = [](double p_area, double p_Ixx, double p_Iyy, double &r_a, double &r_b) {
		r_a = 0.5 * std::sqrt(MAX(12.0 * p_Iyy / p_area, 0.0));
		r_b = 0.5 * std::sqrt(MAX(12.0 * p_Ixx / p_area, 0.0));
	};
	double ha, hb;
	half_dims(A_act, Ixx_a, Iyy_a, ha, hb);
	const double ocx = om.Sx / om.A, ocy = om.Sy / om.A;
	double oa, ob;
	half_dims(om.A, om.Syy - om.A * ocy * ocy, om.Sxx - om.A * ocx * ocx, oa, ob);
	const double half[2][2] = { { ha / h, hb / h }, { oa / h, ob / h } }; // [free, ind][along x, along y]
	double wps[2];
	for (int t = 0; t < 2; ++t) {
		const double lo = MIN(half[t][0], half[t][1]), hi = MAX(half[t][0], half[t][1]);
		wps[t] = 1.0 - lo / hi;
	}
	r_law.A_act = A_act;

	// The friction share: the boundary of the no-slip contact set, each face by its kind (beside a cell outside the soft
	// face: a free face; beside one the partner does not cover: an indenter edge, by the inset tables at its cell's margin;
	// beside a contactable one: a smooth edge)
	double per_free[2] = { 0.0, 0.0 }; // the free faces' lengths, normal along x and along y
	double acc_ind = 0.0, per_ind = 0.0, per_smooth = 0.0;
	for (int k = 0; k < g.n; ++k) {
		if (!ns.active[k]) {
			continue;
		}
		const int i = g.ix(k), j = g.jy(k);
		for (int side = 0; side < 4; ++side) {
			const int axis = side < 2 ? 0 : 1;
			const int ni = side == 0 ? i - 1 : (side == 1 ? i + 1 : i);
			const int nj = side == 2 ? j - 1 : (side == 3 ? j + 1 : j);
			const bool inside = ni >= 0 && ni < g.nx && nj >= 0 && nj < g.ny;
			const int kn = inside ? g.k(ni, nj) : -1;
			const bool nb_dom = inside && w.dom[kn];
			const bool nb_cont = inside && w.cont[kn];
			const bool nb_act = inside && ns.active[kn];
			if (nb_act) {
				continue;
			}
			const double L = axis == 0 ? r_law.DY[k] : r_law.DX[k];
			if (!nb_dom) {
				per_free[axis] += L;
			} else if (!nb_cont) {
				per_ind += L;
				acc_ind += L * f3_inset_phi(half[1][axis], w.marg[k], p_mu, w.kg, wps[1]);
			} else {
				per_smooth += L;
			}
		}
	}
	const double tot_len = per_free[0] + per_free[1] + per_ind + per_smooth;
	double Phi = 1.0;
	if (tot_len > 0.0) {
		double acc = acc_ind;
		for (int axis = 0; axis < 2; ++axis) {
			if (per_free[axis] > 0.0) {
				acc += per_free[axis] * f2_phi(0, half[0][axis], p_mu, w.kg, wps[0], 0.0);
			}
		}
		const double tau = dmax > 1e-300 ? gn * h / dmax : 1e9;
		acc += per_smooth * f2_phi(2, 1.0, p_mu, w.kg, 0.0, tau);
		Phi = MIN(MAX(acc / tot_len, 0.0), 1.0);
	}
	r_law.Phi = Phi;
	r_law.wts[0] = 1.0 - Phi;
	r_law.wts[1] = Phi;
	// A field whose weight is nothing is not solved (the frictionless field of a bonded face)
	if (r_law.wts[0] > 1e-12) {
		if (!f2_solve_field(r_law, w, 0, p_tail)) {
			return false;
		}
	} else {
		r_law.wts[0] = 0.0;
		lub.any = false;
		lub.passes = 0;
		lub.settled = true;
		for (int k = 0; k < g.n; ++k) {
			lub.active[k] = false;
			lub.p[k] = 0.0;
		}
	}

	// The FE edge multipliers (rules 1 and 2): the free face's and the indenter edge's, a cell near both of small margin
	// taking them as one free face by the soft choice
	auto reaches = [&](const bool *p_active, int p_type) -> bool {
		for (int k = 0; k < g.n; ++k) {
			if (p_active[k] && w.e[p_type][k] <= 0.75 * w.ext[p_type][k] + 1e-9 * h) {
				return true;
			}
		}
		return false;
	};
	auto size_of = [&](int p_type, int p_k) -> double { return w.anx[p_type][p_k] * half[p_type][0] + w.any[p_type][p_k] * half[p_type][1]; };
	const double smin = F3_SOFTMIN * h;
	// p_kind 0 lub, 1 ns (the field's multiplier), -1 the friction reshaping (mix)
	auto cell_mult = [&](int p_kind, int p_k, bool p_free, bool p_ind) -> double {
		const bool have_f = p_free && std::isfinite(w.e[0][p_k]);
		const bool have_i = p_ind && std::isfinite(w.e[1][p_k]);
		const double b = have_i ? w.chi[p_k] : 1.0;
		double a;
		if (have_f && have_i) {
			a = 1.0 / (1.0 + std::exp(MIN(MAX((w.e[0][p_k] - w.e[1][p_k]) / smin, -50.0), 50.0)));
		} else {
			a = have_f ? 1.0 : 0.0;
		}
		const double a_f = 1.0 - (1.0 - b) * (1.0 - a);
		const double a_i = 1.0 - (1.0 - b) * a;
		double v = 1.0;
		if (have_f) {
			const double f0 = w.recv_d[p_k] ? 0.0 : MAX(w.e[0][p_k] - 0.5 * w.ext[0][p_k], 0.0) / h;
			const double f1 = (w.e[0][p_k] + 0.5 * w.ext[0][p_k]) / h;
			const double mf = p_kind < 0 ? f2_mix_multiplier(0, size_of(0, p_k), p_mu, f0, f1, wps[0]) : f2_multiplier(p_kind, 0, size_of(0, p_k), f0, f1, wps[0]);
			v *= std::pow(mf, a_f);
		}
		if (have_i) {
			const double i0 = w.recv_c[p_k] ? 0.0 : MAX(w.e[1][p_k] - 0.5 * w.ext[1][p_k], 0.0) / h;
			const double i1 = (w.e[1][p_k] + 0.5 * w.ext[1][p_k]) / h;
			double mi;
			if (p_kind < 0) {
				mi = (1.0 - b) * f2_mix_multiplier(0, size_of(1, p_k), p_mu, i0, i1, wps[1]) + b * f2_mix_multiplier(1, size_of(1, p_k), p_mu, i0, i1, wps[1]);
			} else {
				mi = f3_inset_multiplier(p_kind, size_of(1, p_k), w.marg[p_k], i0, i1, wps[1]);
			}
			v *= std::pow(mi, a_i);
		}
		return v;
	};
	for (int kind = 0; kind < 2; ++kind) {
		F2Field &f = r_law.field[kind];
		for (int k = 0; k < g.n; ++k) {
			w.M[kind][k] = 1.0;
		}
		if (!f.solved) {
			continue;
		}
		const bool pf = reaches(f.active, 0), pi = reaches(f.active, 1);
		for (int k = 0; k < g.n; ++k) {
			if (f.active[k]) {
				w.M[kind][k] = cell_mult(kind, k, pf, pi);
			}
		}
	}
	for (int k = 0; k < g.n; ++k) {
		w.pc[k] = 0.0;
		for (int kind = 0; kind < 2; ++kind) {
			if (r_law.field[kind].solved) {
				w.pc[k] += r_law.wts[kind] * w.M[kind][k] * r_law.field[kind].p[k];
			}
		}
		w.R[k] = 1.0;
	}
	// The friction reshaping at free faces and indenter edges for a Coulomb face (identity at mu 0 and bonded), the edge
	// types the no-slip contact reaches
	if (p_mu > 0.0 && p_mu < double(ftab::MU[ftab::NUM_MU - 1])) {
		const bool pf = reaches(ns.active, 0), pi = reaches(ns.active, 1);
		for (int k = 0; k < g.n; ++k) {
			if (lub.active[k] || ns.active[k]) {
				w.R[k] = cell_mult(-1, k, pf, pi);
			}
		}
		for (int k = 0; k < g.n; ++k) {
			w.pc[k] *= w.R[k];
			w.M[0][k] *= w.R[k];
			w.M[1][k] *= w.R[k];
		}
	}

	// The figures in the caller's frame, each cell standing for its area under the partner
	const double cr = std::cos(r_law.rot), sn = std::sin(r_law.rot);
	double F = 0.0, sx = 0.0, sy = 0.0;
	for (int k = 0; k < g.n; ++k) {
		const double fc = w.pc[k] * w.Ac[k];
		F += fc;
		sx += fc * (cr * r_law.X[k] - sn * r_law.Y[k]);
		sy += fc * (sn * r_law.X[k] + cr * r_law.Y[k]);
	}
	if (!(F > 0.0)) {
		return false;
	}
	const double cx = sx / F, cy = sy / F;
	double rs = 0.0, m1x = 0.0, m1y = 0.0;
	for (int k = 0; k < g.n; ++k) {
		const double fc = w.pc[k] * w.Ac[k];
		const double xo = cr * r_law.X[k] - sn * r_law.Y[k] - cx;
		const double yo = sn * r_law.X[k] + cr * r_law.Y[k] - cy;
		rs += fc * std::hypot(xo, yo);
		m1x += fc * std::abs(xo);
		m1y += fc * std::abs(yo);
	}
	r_law.F = F;
	r_law.cx = cx;
	r_law.cy = cy;
	r_law.r_s = rs / F;
	r_law.m1x = m1x / F;
	r_law.m1y = m1y / F;
	double area = 0.0, Ixx = 0.0, Iyy = 0.0;
	r_law.edge_x[0] = r_law.edge_x[1] = r_law.edge_y[0] = r_law.edge_y[1] = 0.0;
	for (int kind = 0; kind < 2; ++kind) {
		const F2Field &f = r_law.field[kind];
		if (!f.solved) {
			continue;
		}
		double fa = 0.0, fxx = 0.0, fyy = 0.0;
		double ex0 = 1e300, ex1 = -1e300, ey0 = 1e300, ey1 = -1e300;
		for (int k = 0; k < g.n; ++k) {
			if (!f.active[k]) {
				continue;
			}
			const double xo = cr * r_law.X[k] - sn * r_law.Y[k];
			const double yo = sn * r_law.X[k] + cr * r_law.Y[k];
			const double DX2 = r_law.DX[k] * r_law.DX[k], DY2 = r_law.DY[k] * r_law.DY[k];
			fa += w.Ac[k];
			fyy += w.Ac[k] * ((yo - cy) * (yo - cy) + (DX2 * sn * sn + DY2 * cr * cr) / 12.0);
			fxx += w.Ac[k] * ((xo - cx) * (xo - cx) + (DX2 * cr * cr + DY2 * sn * sn) / 12.0);
			const double hx = 0.5 * (std::abs(cr) * r_law.DX[k] + std::abs(sn) * r_law.DY[k]);
			const double hy = 0.5 * (std::abs(sn) * r_law.DX[k] + std::abs(cr) * r_law.DY[k]);
			ex0 = MIN(ex0, xo - hx);
			ex1 = MAX(ex1, xo + hx);
			ey0 = MIN(ey0, yo - hy);
			ey1 = MAX(ey1, yo + hy);
		}
		area += r_law.wts[kind] * fa;
		Ixx += r_law.wts[kind] * fyy;
		Iyy += r_law.wts[kind] * fxx;
		if (f.any) {
			r_law.edge_x[0] += r_law.wts[kind] * (ex0 - cx);
			r_law.edge_x[1] += r_law.wts[kind] * (ex1 - cx);
			r_law.edge_y[0] += r_law.wts[kind] * (ey0 - cy);
			r_law.edge_y[1] += r_law.wts[kind] * (ey1 - cy);
		}
	}
	r_law.area = area;
	r_law.Ixx = Ixx;
	r_law.Iyy = Iyy;
	r_law.J_p = Ixx + Iyy;

	// The normal and tilt stiffnesses: each field's tangent pressures about the centre (in the grid's frame), times its
	// multipliers, weight and contact area; the tilt tensor turned back
	const double cxg = cr * cx + sn * cy, cyg = -sn * cx + cr * cy;
	double kn = 0.0, kxx = 0.0, kyy = 0.0, kxy = 0.0;
	for (int kind = 0; kind < 2; ++kind) {
		const F2Field &f = r_law.field[kind];
		if (!f.solved) {
			continue;
		}
		for (int k = 0; k < g.n; ++k) {
			const double s = w.M[kind][k] * r_law.wts[kind] * w.Ac[k];
			const double p1 = f.t1[k] * s;
			const double px = (f.tx[k] - cxg * f.t1[k]) * s;
			const double py = (f.ty[k] - cyg * f.t1[k]) * s;
			kn += p1;
			kxx += px * (r_law.X[k] - cxg);
			kxy += px * (r_law.Y[k] - cyg);
			kyy += py * (r_law.Y[k] - cyg);
		}
	}
	r_law.k_n = kn;
	// R K R^T with R = [[cr, -sn], [sn, cr]]
	const double K00 = cr * (cr * kxx - sn * kxy) - sn * (cr * kxy - sn * kyy);
	const double K01 = cr * (sn * kxx + cr * kxy) - sn * (sn * kxy + cr * kyy);
	const double K11 = sn * (sn * kxx + cr * kxy) + cr * (sn * kxy + cr * kyy);
	r_law.kt_y = K00;
	r_law.kt_x = K11;
	r_law.kt_xy = K01;
	r_law.passes = MAX(lub.passes, ns.passes);
	r_law.settled = lub.settled && ns.settled;
	// The held slide's memory after this step (rule 6): new contact area takes the current gradient, area weighted; a
	// shrinking contact keeps the gradient and its smaller area, so that area regained counts as new
	if (!has_held) {
		r_law.held_out[0] = p_gx;
		r_law.held_out[1] = p_gy;
	} else if (A_act > p_tail->held[2]) {
		const double wt = p_tail->held[2] / A_act;
		r_law.held_out[0] = wt * p_tail->held[0] + (1.0 - wt) * p_gx;
		r_law.held_out[1] = wt * p_tail->held[1] + (1.0 - wt) * p_gy;
	} else {
		r_law.held_out[0] = p_tail->held[0];
		r_law.held_out[1] = p_tail->held[1];
	}
	r_law.held_out[2] = A_act;
	r_law.empty = false;
	return true;
}

// The distance from the centre to the contact's edge along an in-plane direction (mx, my) of the caller's frame: each
// field's, weighted as the figures are
double f2_edge_distance(const F2Law &p_law, double p_mx, double p_my) {
	const double cr = std::cos(p_law.rot), sn = std::sin(p_law.rot);
	// the direction in the grid's frame
	const double gmx = cr * p_mx + sn * p_my, gmy = -sn * p_mx + cr * p_my;
	double out = 0.0;
	for (int kind = 0; kind < 2; ++kind) {
		const F2Field &f = p_law.field[kind];
		if (!f.solved || !f.any) {
			continue;
		}
		double best = 0.0;
		for (int k = 0; k < p_law.grid.n; ++k) {
			if (f.active[k]) {
				const double xo = cr * p_law.X[k] - sn * p_law.Y[k] - p_law.cx;
				const double yo = sn * p_law.X[k] + cr * p_law.Y[k] - p_law.cy;
				best = MAX(best, xo * p_mx + yo * p_my + 0.5 * (p_law.DX[k] * std::abs(gmx) + p_law.DY[k] * std::abs(gmy)));
			}
		}
		out += p_law.wts[kind] * best;
	}
	return out;
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

// The logs of the layer table's log axes' nodes, made once (as F2AxisLogs): the same std::log of the same floats
// axis_weights, kg_cell and the profiles' cells took on every lookup
struct LayerAxisLogs {
	double rh[table::NUM_RH];
	double sh[table::NUM_SH];
	double kg[table::NUM_KG];
	double t3[table::T3_NUM_S];
	double t4[table::T4_NUM_C];

	LayerAxisLogs() {
		for (int i = 0; i < table::NUM_RH; ++i) {
			rh[i] = std::log(double(table::RH[i]));
		}
		for (int i = 0; i < table::NUM_SH; ++i) {
			sh[i] = std::log(double(table::SH[i]));
		}
		for (int i = 0; i < table::NUM_KG; ++i) {
			kg[i] = std::log(double(table::KG[i]));
		}
		for (int i = 0; i < table::T3_NUM_S; ++i) {
			t3[i] = std::log(double(table::T3_S[i]));
		}
		for (int i = 0; i < table::T4_NUM_C; ++i) {
			t4[i] = std::log(double(table::T4_C[i]));
		}
	}
};

const LayerAxisLogs &layer_axis_logs() {
	static const LayerAxisLogs logs;
	return logs;
}

// The node logs of a layer table axis, or nullptr for an axis it does not hold
const double *layer_axis_nodes(const float *p_axis, int p_n) {
	const LayerAxisLogs &logs = layer_axis_logs();
	if (p_axis == table::RH && p_n == table::NUM_RH) {
		return logs.rh;
	}
	if (p_axis == table::SH && p_n == table::NUM_SH) {
		return logs.sh;
	}
	if (p_axis == table::KG && p_n == table::NUM_KG) {
		return logs.kg;
	}
	if (p_axis == table::T3_S && p_n == table::T3_NUM_S) {
		return logs.t3;
	}
	if (p_axis == table::T4_C && p_n == table::T4_NUM_C) {
		return logs.t4;
	}
	return nullptr;
}

// Weights on the nodes of one log axis at p_value and their derivatives with respect to its logarithm (4.2)
AxisWeights axis_weights(const float *p_axis, int p_n, double p_value, int p_order) {
	double own[16];
	const double *x = layer_axis_nodes(p_axis, p_n);
	if (x == nullptr) {
		for (int k = 0; k < p_n && k < 16; ++k) {
			own[k] = std::log(double(p_axis[k]));
		}
		x = own;
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
	const double *lkg = layer_axis_logs().kg;
	int k = -1;
	for (int i = 0; i < table::NUM_KG; ++i) {
		if (lkg[i] <= X) {
			k = i;
		}
	}
	k = MIN(MAX(k, 0), table::NUM_KG - 2);
	const double x0 = lkg[k], x1 = lkg[k + 1];
	r_k = k;
	r_t = MIN(MAX((X - x0) / (x1 - x0), 0.0), 1.0);
}

struct Lookup {
	double force = 0.0; // CYL: P' / (E h); SPH: F / (E h^2)
	double width = 0.0; // CYL: b / h; SPH: a / h
	double m1 = 0.0; // m1 / h
	double spin = 0.0; // SPH: r_s / h
	double slope = 0.0; // d ln F / d ln squash
	double rslope = 0.0; // d ln F / d ln (R / h), for two curved soft sides (Stage 1c-b fix round, C3)
	double profile[table::NUM_ST];
	int order = 3;
};

// The logs of the layer table's positive scalars, made once (as F2AxisLogs): the same std::log of the same floats
// scalar() took for every node of its stencil on every lookup, about 290 a lookup
struct LayerScalarLogs {
	typedef double Grid[table::NUM_KG][table::NUM_FACE][table::NUM_RH][table::NUM_SH];
	Grid cyl_force, cyl_halfwidth, cyl_m1, sph_force, sph_radius, sph_m1, sph_spin;

	static void fill(const float (*p_arr)[table::NUM_FACE][table::NUM_RH][table::NUM_SH], Grid &r_logs) {
		for (int k = 0; k < table::NUM_KG; ++k) {
			for (int a = 0; a < table::NUM_FACE; ++a) {
				for (int b = 0; b < table::NUM_RH; ++b) {
					for (int c = 0; c < table::NUM_SH; ++c) {
						r_logs[k][a][b][c] = std::log(double(p_arr[k][a][b][c]));
					}
				}
			}
		}
	}

	LayerScalarLogs() {
		fill(table::CYL_FORCE, cyl_force);
		fill(table::CYL_HALFWIDTH, cyl_halfwidth);
		fill(table::CYL_M1, cyl_m1);
		fill(table::SPH_FORCE, sph_force);
		fill(table::SPH_RADIUS, sph_radius);
		fill(table::SPH_M1, sph_m1);
		fill(table::SPH_SPIN, sph_spin);
	}
};

const LayerScalarLogs &layer_scalar_logs() {
	static const LayerScalarLogs logs;
	return logs;
}

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

	// p_logs: the scalar's logs (LayerScalarLogs)
	auto scalar = [&](const double (*p_logs)[table::NUM_FACE][table::NUM_RH][table::NUM_SH], bool p_slope, double *r_slope, double *r_rslope) {
		double acc = 0.0, dacc = 0.0, racc = 0.0;
		for (int k = 0; k < wk.count; ++k) {
			for (int a = 0; a < 2; ++a) {
				for (int b = 0; b < wu.count; ++b) {
					for (int c = 0; c < wv.count; ++c) {
						const double lq = p_logs[wk.index[k]][wf_index[a]][wu.index[b]][wv.index[c]];
						acc += wk.w[k] * wf[a] * wu.w[b] * wv.w[c] * lq;
						if (p_slope) {
							dacc += wk.w[k] * wf[a] * wu.w[b] * wv.dw[c] * lq;
							racc += wk.w[k] * wf[a] * wu.dw[b] * wv.w[c] * lq;
						}
					}
				}
			}
		}
		if (r_slope != nullptr) {
			*r_slope = dacc;
		}
		if (r_rslope != nullptr) {
			*r_rslope = racc;
		}
		return std::exp(acc);
	};
	const LayerScalarLogs &logs = layer_scalar_logs();
	if (p_cyl) {
		r_out.force = scalar(logs.cyl_force, true, &r_out.slope, &r_out.rslope);
		r_out.width = scalar(logs.cyl_halfwidth, false, nullptr, nullptr);
		r_out.m1 = scalar(logs.cyl_m1, false, nullptr, nullptr);
		r_out.spin = 0.0;
	} else {
		r_out.force = scalar(logs.sph_force, true, &r_out.slope, &r_out.rslope);
		r_out.width = scalar(logs.sph_radius, false, nullptr, nullptr);
		r_out.m1 = scalar(logs.sph_m1, false, nullptr, nullptr);
		r_out.spin = scalar(logs.sph_spin, false, nullptr, nullptr);
	}

	// Profiles: bilinear in the two logs with both cell coordinates clamped, linear in the face and in ln K/G
	const int i = wu.cell;
	const int j = wv.cell;
	const LayerAxisLogs &axis_logs = layer_axis_logs();
	double lu = (std::log(p_rh) - axis_logs.rh[i]) / (axis_logs.rh[i + 1] - axis_logs.rh[i]);
	double lv = (std::log(p_sh) - axis_logs.sh[j]) / (axis_logs.sh[j + 1] - axis_logs.sh[j]);
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

// Two curved soft sides through the table (Stage 1c-b fix round, C3; the desk reference's section 9b): the lead's
// decision 5, each side's layer with a lubricated face (the table's mu 0 column) and the two compliances in series. Side i
// takes the share phi_i of the pair's squash s and sees the other side as a radius R / phi_i: where the layers act
// locally the interface pressure is p = (s - r^2 / 2R) / (1/k_1 + 1/k_2), so side i's compression is phi_i s - r^2 /
// (2 R / phi_i). phi_1 makes the two sides' forces equal (a safeguarded Newton on ln F_1 - ln F_2 from the local terms'
// share h_1/E_1 over the pair's). Exact for two identical sides: by the mirror symmetry the interface carries no shear and
// each side carries half the gap (phi 1/2, radius 2R). Against fresh two-body FE (two layers, a frictionless interface)
// within 1.3 % at a Point and 1.1 % along a Line (stage1cb/c3/fe_two.out.txt). The figures other than the force are the
// side with the larger share's (side 1 on a tie).
constexpr int TWO_SIDED_NEWTON_CAP = 40; // Newton (or bisection) steps on the share
constexpr double TWO_SIDED_F_TOL = 1e-12; // |ln F_1 - ln F_2| at which the share is taken
constexpr double TWO_SIDED_PHI_TOL = 1e-14; // the bracket's width at which it is not (the forces still differ)

// The local terms' share of side 1, h_1/E_1 over the pair's: Newton's start (exactly 1/2 for identical sides)
double two_sided_share0(const JoltSoftSurface3D &p_s1, const JoltSoftSurface3D &p_s2) {
	const double a = p_s1.thickness / p_s1.modulus;
	const double b = p_s2.thickness / p_s2.modulus;
	return a / (a + b);
}

// The share phi of side 1 at the pair's squash p_s against a radius p_R (p_cyl: a Line's slice, per unit length), and the
// two sides' lookups at their shares. False outside the table (a side's lookup outside it anywhere the share can lie, or
// no share with equal forces inside it).
bool two_sided_split(bool p_cyl, double p_s, double p_R, const JoltSoftSurface3D &p_s1, const JoltSoftSurface3D &p_s2, double &r_phi, Lookup &r_l1, Lookup &r_l2) {
	if (!(p_s > 0.0)) {
		return false;
	}
	const double rh0 = double(table::RH[0]);
	const double rh1 = double(table::RH[table::NUM_RH - 1]);
	const double sh1 = double(table::SH[table::NUM_SH - 1]);
	const double h1 = p_s1.thickness;
	const double h2 = p_s2.thickness;
	double lo = MAX(MAX(p_R / (h1 * rh1), 1.0 - p_R / (h2 * rh0)), MAX(1.0 - sh1 * h2 / p_s, 1e-9));
	double hi = MIN(MIN(p_R / (h1 * rh0), sh1 * h1 / p_s), MIN(1.0 - p_R / (h2 * rh1), 1.0 - 1e-9));
	if (!(lo < hi)) {
		return false;
	}
	double phi = MIN(MAX(two_sided_share0(p_s1, p_s2), lo), hi);
	const double sc1 = p_cyl ? p_s1.modulus * h1 : p_s1.modulus * h1 * h1;
	const double sc2 = p_cyl ? p_s2.modulus * h2 : p_s2.modulus * h2 * h2;
	const double kg1 = kg_of(p_s1);
	const double kg2 = kg_of(p_s2);
	for (int it = 0; it < TWO_SIDED_NEWTON_CAP; ++it) {
		const double q = 1.0 - phi;
		if (!table_lookup(p_cyl, p_R / (phi * h1), phi * p_s / h1, 0.0, kg1, r_l1) || !table_lookup(p_cyl, p_R / (q * h2), q * p_s / h2, 0.0, kg2, r_l2)) {
			return false;
		}
		if (!(r_l1.force > 0.0) || !(r_l2.force > 0.0)) {
			return false;
		}
		const double f = std::log(sc1 * r_l1.force) - std::log(sc2 * r_l2.force);
		if (std::abs(f) <= TWO_SIDED_F_TOL) {
			r_phi = phi;
			return true;
		}
		if (f < 0.0) {
			lo = phi;
		} else {
			hi = phi;
		}
		if (hi - lo <= TWO_SIDED_PHI_TOL) {
			return false;
		}
		const double c1 = r_l1.slope - r_l1.rslope;
		const double c2 = r_l2.slope - r_l2.rslope;
		const double d = c1 / phi + c2 / q;
		double nxt = d > 0.0 ? phi - f / d : 0.5 * (lo + hi);
		if (!(lo < nxt && nxt < hi)) {
			nxt = 0.5 * (lo + hi);
		}
		phi = nxt;
	}
	return false;
}

// n = d ln F / d ln s of the pair at its share: (c1 c2 + phi_2 c1 b2 + phi_1 c2 b1) / (phi_2 c1 + phi_1 c2), a_i the table's
// slope in the squash, b_i its slope in the radius, c_i = a_i - b_i (n = a for identical sides, 2 where the layers act
// locally)
double two_sided_exponent(double p_phi, const Lookup &p_l1, const Lookup &p_l2) {
	const double q = 1.0 - p_phi;
	const double b1 = p_l1.rslope;
	const double b2 = p_l2.rslope;
	const double c1 = p_l1.slope - b1;
	const double c2 = p_l2.slope - b2;
	const double den = q * c1 + p_phi * c2;
	if (!(den > 0.0)) {
		return p_phi >= 0.5 ? p_l1.slope : p_l2.slope;
	}
	return (c1 * c2 + q * c1 * b2 + p_phi * c2 * b1) / den;
}

// p_layer2 (Stage 1c-b fix round, C3): two curved soft sides, each slice through two_sided_split (p_mu unused)
bool table_line(double p_s0, double p_camber, double p_ya, double p_yb, double p_R, const JoltSoftSurface3D &p_layer, double p_mu, CurvedFig &r_fig, const JoltSoftSurface3D *p_layer2 = nullptr) {
	const double h = p_layer.thickness;
	const double E = p_layer.modulus;
	const double kg = kg_of(p_layer);
	const double dy = (p_yb - p_ya) / N_SLICES;
	double ys[N_SLICES], P[N_SLICES], b[N_SLICES], m1[N_SLICES], dP[N_SLICES], prof[N_SLICES][table::NUM_ST];
	// A slice at the squash of the last slice looked up takes that lookup again: the same arguments into the same pure
	// lookup, which writes every member read below. A level strip (camber 0, as the first-touch law's reference) has one
	// squash on every slice, so it makes one lookup, not 24.
	Lookup look;
	double look_squash = 0.0;
	bool look_have = false;
	for (int i = 0; i < N_SLICES; ++i) {
		ys[i] = p_ya + (double(i) + 0.5) * dy;
		P[i] = b[i] = m1[i] = dP[i] = 0.0;
		const double s = p_s0 + p_camber * ys[i];
		if (!(s > 0.0)) {
			continue;
		}
		if (p_layer2 != nullptr) {
			double phi = 0.0;
			Lookup l1, l2;
			if (!two_sided_split(true, s, p_R, p_layer, *p_layer2, phi, l1, l2)) {
				return false;
			}
			const bool first = phi >= 0.5;
			const Lookup &r = first ? l1 : l2;
			const JoltSoftSurface3D &L = first ? p_layer : *p_layer2;
			P[i] = L.modulus * L.thickness * r.force;
			b[i] = L.thickness * r.width;
			m1[i] = L.thickness * r.m1;
			dP[i] = P[i] * two_sided_exponent(phi, l1, l2) / s;
			for (int k = 0; k < table::NUM_ST; ++k) {
				prof[i][k] = r.profile[k];
			}
			continue;
		}
		if (!look_have || s != look_squash) {
			if (!table_lookup(true, p_R / h, s / h, p_mu, kg, look)) {
				return false;
			}
			look_squash = s;
			look_have = true;
		}
		P[i] = E * h * look.force;
		b[i] = h * look.width;
		m1[i] = h * look.m1;
		dP[i] = P[i] * look.slope / s;
		for (int k = 0; k < table::NUM_ST; ++k) {
			prof[i][k] = look.profile[k];
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

// A Point's figures (as table_point) for two curved soft sides through the table. False outside it.
bool two_sided_point(double p_s, double p_R, const JoltSoftSurface3D &p_s1, const JoltSoftSurface3D &p_s2, CurvedFig &r_fig) {
	double phi = 0.0;
	Lookup l1, l2;
	if (!two_sided_split(false, p_s, p_R, p_s1, p_s2, phi, l1, l2)) {
		return false;
	}
	const bool first = phi >= 0.5;
	const Lookup &r = first ? l1 : l2;
	const JoltSoftSurface3D &L = first ? p_s1 : p_s2;
	const double h = L.thickness;
	const double F = L.modulus * h * h * r.force;
	const double a = h * r.width;
	const double n = two_sided_exponent(phi, l1, l2);
	r_fig.empty = !(F > 0.0);
	r_fig.F = F;
	r_fig.cx = r_fig.cy = 0.0;
	r_fig.r_s = h * r.spin;
	r_fig.m1x = r_fig.m1y = h * r.m1;
	r_fig.k_n = F * n / p_s;
	r_fig.area = PI * a * a;
	r_fig.J_p = 0.5 * PI * a * a * a * a;
	r_fig.sigma = a;
	r_fig.n_exp = n;
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

// Two curved soft sides where the series cannot be read through the table (a side outside it; Stage 1c-b fix round, C3):
// the side with the larger local share phi (h/E over the pair's; side 1 on a tie) reads its own single-layer law with a
// lubricated face at the squash phi s0, the radius R / phi and the camber phi camber: the table at mu 0 where that side is
// in it, else its fallback (the split law on the grid, T3 at mu 0). Its normal and camber stiffness are per the pair's
// squash (times phi). Counted as outside the table. Exact for identical sides as the series is; against the two-body FE
// with a 70 A side (outside the table's K / G axis) within 1.1 % (stage1cb/c3/fe_two.out.txt).
bool two_sided_outside(bool p_line, double p_s0, double p_R, const JoltSoftSurface3D &p_s1, const JoltSoftSurface3D &p_s2, double p_camber, double p_ya, double p_yb, CurvedFig &r_fig) {
	const double p1 = two_sided_share0(p_s1, p_s2);
	const bool first = p1 >= 0.5;
	const double phi = first ? p1 : 1.0 - p1;
	const JoltSoftSurface3D &L = first ? p_s1 : p_s2;
	const bool in_table = p_line ? table_line(phi * p_s0, phi * p_camber, p_ya, p_yb, p_R / phi, L, 0.0, r_fig) : table_point(phi * p_s0, p_R / phi, L, 0.0, r_fig);
	if (in_table) {
		if (r_fig.empty || !(r_fig.F > 0.0)) {
			r_fig.empty = true;
			return true;
		}
	} else {
		const Coeffs co = split_coeffs(L.modulus, L.bulk_modulus, L.thickness, false);
		if (!curved_outside(p_line, phi * p_s0, p_R / phi, co, phi * p_camber, p_ya, p_yb, r_fig)) {
			return false;
		}
		if (r_fig.empty || !(r_fig.F > 0.0) || !(r_fig.area > 0.0)) {
			r_fig.empty = true;
			return true;
		}
		// a second moment Richardson takes below zero (a contact the grid barely resolves) reads T3's lowest node
		double S = shape_factor(r_fig.area, r_fig.Ixx, r_fig.Iyy, L.thickness);
		if (!(S > 0.0)) {
			S = 0.0;
		}
		r_fig.face_ratio = table_ratio(false, S, 0.0, kg_of(L));
		r_fig.F *= r_fig.face_ratio;
		r_fig.k_n *= r_fig.face_ratio;
		r_fig.kt_x *= r_fig.face_ratio;
	}
	r_fig.k_n *= phi;
	r_fig.kt_x *= phi;
	r_fig.outside_table = true;
	return true;
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

// A supporting face's plane normal (Newell's, from its vertices), turned to the side of p_toward; zero for a face with
// fewer than 3 vertices, a degenerate one, or one more than 60 deg from p_toward (then it is not the face p_toward found)
JPH::Vec3 face_normal(const SupportFace &p_face, const JPH::Vec3 &p_toward) {
	if (p_face.n < 3) {
		return JPH::Vec3::sZero();
	}
	double nx = 0.0, ny = 0.0, nz = 0.0;
	for (int k = 0; k < p_face.n; ++k) {
		const JPH::Vec3 &a = p_face.v[k];
		const JPH::Vec3 &b = p_face.v[(k + 1) % p_face.n];
		nx += (double(a.GetY()) - double(b.GetY())) * (double(a.GetZ()) + double(b.GetZ()));
		ny += (double(a.GetZ()) - double(b.GetZ())) * (double(a.GetX()) + double(b.GetX()));
		nz += (double(a.GetX()) - double(b.GetX())) * (double(a.GetY()) + double(b.GetY()));
	}
	const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
	if (!(len > 0.0)) {
		return JPH::Vec3::sZero();
	}
	JPH::Vec3 fn(float(nx / len), float(ny / len), float(nz / len));
	const float c = fn.Dot(p_toward);
	if (std::abs(c) < 0.5f) {
		return JPH::Vec3::sZero();
	}
	return c < 0.0f ? -fn : fn;
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

// The change of the gap between two faces' planes across their footprints' overlap, along p_n (the soft face's own
// normal, so a manifold normal that sits off two parallel faces for a step cannot move it): -1 when the footprints do not
// overlap, infinity when either plane is more than about 78 deg from p_n (face_plane refuses it). The desk's
// soft_ref.face_spread (section 10b).
double face_spread(const SupportFace &p_soft, const SupportFace &p_partner, const JPH::Vec3 &p_n) {
	const JPH::Vec3 t1 = p_n.GetNormalizedPerpendicular();
	const JPH::Vec3 t2 = p_n.Cross(t1);
	const JPH::Vec3 o = JPH::Vec3::sZero();
	const HeightPlane ps = face_plane(p_soft, o, t1, t2, p_n);
	const HeightPlane pp = face_plane(p_partner, o, t1, t2, p_n);
	if (!ps.valid || !pp.valid) {
		return std::numeric_limits<double>::infinity();
	}
	Poly overlap;
	clip_convex(project(p_soft, o, t1, t2), project(p_partner, o, t1, t2), overlap);
	if (overlap.n < 3) {
		return -1.0;
	}
	const double gx = pp.gx - ps.gx;
	const double gy = pp.gy - ps.gy;
	double lo = std::numeric_limits<double>::max();
	double hi = -std::numeric_limits<double>::max();
	for (int k = 0; k < overlap.n; ++k) {
		const double g = gx * overlap.x[k] + gy * overlap.y[k];
		lo = MIN(lo, g);
		hi = MAX(hi, g);
	}
	return hi - lo;
}

// An edge face's deepest edge into a flat face along the manifold's normal p_n: its deepest vertex and the deeper of that
// vertex's two neighbours (the later one on a tie), and how much shallower that neighbour is (r_drop). Body 1's vertices go
// past the flat face along +p_n, body 2's along -p_n. False for an edge face of fewer than 3 vertices. The desk's
// soft_ref.deepest_edge (section 10b).
bool deepest_edge(const SupportFace &p_edge_face, const SupportFace &p_flat, const JPH::Vec3 &p_n, bool p_edge_is_1, SupportFace &r_edge, double &r_drop) {
	const int m = p_edge_face.n;
	if (m < 3 || p_flat.n < 1) {
		return false;
	}
	double pen[32];
	int k = 0;
	for (int i = 0; i < m; ++i) {
		const double d = double((p_edge_face.v[i] - p_flat.v[0]).Dot(p_n));
		pen[i] = p_edge_is_1 ? d : -d;
		if (pen[i] > pen[k]) {
			k = i;
		}
	}
	const int a = (k + m - 1) % m;
	const int b = (k + 1) % m;
	const int j = pen[b] >= pen[a] ? b : a;
	r_edge.n = 2;
	r_edge.v[0] = p_edge_face.v[k];
	r_edge.v[1] = p_edge_face.v[j];
	r_drop = pen[k] - pen[j];
	return true;
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

	JPH::Vec3 n = manifold.mWorldSpaceNormal;
	const JPH::RVec3 base = manifold.mBaseOffset;
	// A face patch keeps to its last step's normal when the manifold's turns more than 45 deg in one step and reads a
	// penetration at least twice the overlap the bodies show along the last one (Stage 1c-b fix round). The manifold's normal
	// is meant to be the direction of least overlap, so a larger one than the last normal's means the penetration search
	// failed for that step, which it can on two boxes with coincident faces (PS3's twist at 3 kHz: a normal 68 deg off with
	// a depth of 13 mm on a 2 mm layer, and the faces picked along it the side faces, a patch 15 mm deep). A contact that
	// truly moves to other faces (a block slid off an edge) turns with no more overlap along its new normal than along the
	// last, and is never held.
	if (p_in.tail != nullptr && manifold.mPenetrationDepth > 0.0f) {
		const JPH::Vec3 prev(p_in.tail->mFaceNormal[0], p_in.tail->mFaceNormal[1], p_in.tail->mFaceNormal[2]);
		const float len = prev.Length();
		if (len > 0.5f && n.Dot(prev / len) < 0.70710678f) {
			const JPH::Vec3 pn = prev / len;
			SupportFace last1, last2;
			supporting_face(body1, manifold.mSubShapeID1, pn, base, last1);
			supporting_face(body2, manifold.mSubShapeID2, -pn, base, last2);
			if (last1.n > 0 && last2.n > 0) {
				// The overlap along the last normal: body 1's furthest point along it less body 2's nearest
				float top = -std::numeric_limits<float>::max();
				float bottom = std::numeric_limits<float>::max();
				for (int k = 0; k < last1.n; ++k) {
					top = MAX(top, last1.v[k].Dot(pn));
				}
				for (int k = 0; k < last2.n; ++k) {
					bottom = MIN(bottom, last2.v[k].Dot(pn));
				}
				if (top - bottom < 0.5f * manifold.mPenetrationDepth) {
					n = pn;
					r_patch.mNormal = n; // the rows take it unless the face's own normal replaces it below
				}
			}
		}
	}

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
	// A Face patch is for two faces pressed flat on each other (the edge lane, the desk's soft_ref section 10b). Two
	// flat faces can be pressed flat on each other only where the layer can take up the tilt between them across the
	// patch: the gap between their planes changes across their footprints' overlap by its spread, and the layer
	// squashes at most its depth (both layers' for two). Beyond it the far side of the face can never reach the other
	// face, and the contact is an edge of one face on the other, built along the manifold's normal as a rigid contact
	// is and as every Line and Point is (a soft TT tyre's cap 30 deg off a wall, built as a Face along the cap's own
	// normal, pushed the tyre back along the wall at its first touch, and along its own motion at an impact). The face
	// is the one whose normal is nearer the manifold's (the direction of least penetration); the edge is the other
	// face's deepest edge. The same test one dimension down: an edge whose far end sits more than the layer's depth
	// shallower than its deep end can only ever press at that end, so the contact is its deepest vertex, a Point (a
	// cylinder's cap: Jolt turns its octagon so a vertex points along the tilt, and at 30 deg its edges climb 4.6 mm
	// over 24 mm); otherwise a Line along the edge. Either is rounded to the layer's depth below (design 4.1 item 2, a
	// polyhedral edge, whatever the leaf: a cylinder's rim is a corner, not its side). The spread is read in the soft
	// face's own frame, so a manifold normal that sits off two parallel faces for a step (PS6's island) leaves them a
	// Face. A Layer leads it: a Lumped surface has no depth to take up a tilt and no rounding for an edge, and a Solid
	// one against a face keeps the rigid manifold.
	SupportFace edge_face;
	bool face_edge = false;
	if (cls == 0 && soft.kind == JoltSoftSurface3D::KIND_LAYER) {
		const SupportFace &soft_face = soft_is_1 ? face1 : face2;
		const SupportFace &partner_face = soft_is_1 ? face2 : face1;
		const JPH::Vec3 ns = face_normal(soft_face, n);
		const JPH::Vec3 np = face_normal(partner_face, n);
		if (ns != JPH::Vec3::sZero() && np != JPH::Vec3::sZero()) {
			const double depth = two_layers ? s1->thickness + s2->thickness : soft.thickness;
			if (face_spread(soft_face, partner_face, ns) > depth) {
				const bool soft_flat = ns.Dot(n) >= np.Dot(n);
				const bool flat_is_1 = soft_flat == soft_is_1;
				double drop = 0.0;
				if (deepest_edge(flat_is_1 ? face2 : face1, flat_is_1 ? face1 : face2, n, !flat_is_1, edge_face, drop)) {
					if (drop > depth) {
						edge_face.n = 1;
					}
					cls = edge_face.n == 2 ? 1 : 2;
					flat = flat_is_1 ? &face1 : &face2;
					curved = &edge_face;
					curved_is_1 = !flat_is_1;
					face_edge = true;
				}
			}
		}
	}
	// A face patch is built and solved along its soft face's own normal (Stage 1c-b gates). The manifold's normal comes from
	// the penetration search, which on a squash of a few tenths of a micrometre between two boxes can sit tens of degrees
	// off the faces for a step (PS6's island at 1 kHz: 28 deg), and the law, its frame and every row would follow it: the
	// pad read foreshortened, its force pushing sideways. The faces are picked again along the face normal; if that leaves
	// either side without a face, the manifold's normal stays.
	if (cls == 0) {
		const JPH::Vec3 fn = face_normal(soft_is_1 ? face1 : face2, n);
		if (fn != JPH::Vec3::sZero()) {
			SupportFace again1, again2;
			supporting_face(body1, manifold.mSubShapeID1, fn, base, again1);
			supporting_face(body2, manifold.mSubShapeID2, -fn, base, again2);
			if (again1.n >= 3 && again2.n >= 3) {
				face1 = again1;
				face2 = again2;
				n = fn;
				r_patch.mNormal = fn;
			}
		}
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
		thread_local F2Law scratch_law_v2;
		F2Law *law2 = &scratch_law_v2;
		// A Layer face against a partner that is not a Layer takes the face law v2 (Stage 1c-b desk A); two Layer faces keep
		// the split law in the lead's decision 5
		const bool face_v2 = soft.kind == JoltSoftSurface3D::KIND_LAYER && !two_layers;
		bool pressed = false;
		if (soft.kind == JoltSoftSurface3D::KIND_LAYER) {
			if (face_v2) {
				F2Tail tail;
				const JPH::SoftContactTail *t = p_in.tail;
				if (t != nullptr && (t->mGridKind == JPH::SoftContactTail::GridFaceV2 || t->mGridKind == JPH::SoftContactTail::GridFaceV2NoLub)) {
					tail.words = t->mActiveSet;
					tail.nx = t->mGridLong;
					tail.ny = t->mGridShort;
					tail.has[0] = t->mGridKind == JPH::SoftContactTail::GridFaceV2;
					tail.has[1] = true;
				}
				// the held slide's memory the last step's face patch left (the face law v3, rule 6)
				if (t != nullptr && t->mHeldArea > 0.0f) {
					tail.has_held = true;
					tail.held[0] = t->mHeldGradient[0];
					tail.held[1] = t->mHeldGradient[1];
					tail.held[2] = t->mHeldArea;
				}
				pressed = f2_face_law(soft_poly, partners, d0, gx, gy, soft.modulus, soft.bulk_modulus, soft.thickness, mu, F2_EDGE_TOL, &tail, *law2) && !law2->empty;
			} else {
				pressed = face_law(soft_poly, partners, d0 * share, gx * share, gy * share, co, p_in.tail, *law) && !law->empty;
			}
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
			double kt_x, kt_y, kt_xy, area, J_p, ex0, ex1;
			if (face_v2) {
				// The face law v2 carries the face condition itself (its two fields and the friction share Phi): no ratio is
				// applied, and the figures report Phi as the face condition
				r_result.passes = law2->passes;
				r_result.not_settled = !law2->settled;
				r_result.raw_force = law2->F;
				kt_x = law2->kt_x;
				kt_y = law2->kt_y;
				kt_xy = law2->kt_xy;
				kn = law2->k_n;
				F = law2->F;
				ratio = law2->Phi;
				cx = law2->cx;
				cy = law2->cy;
				area = law2->area;
				J_p = law2->J_p;
				ex0 = law2->edge_x[0];
				ex1 = law2->edge_x[1];
				r_patch.mSpinRadius = float(law2->r_s);
				r_result.m1[0] = law2->m1x;
				r_result.m1[1] = law2->m1y;
			} else {
				r_result.passes = law->full.passes;
				r_result.not_settled = !law->full.settled;
				r_result.raw_force = law->F;
				kt_x = law->kt_x;
				kt_y = law->kt_y;
				kt_xy = law->kt_xy;
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
				area = law->area;
				J_p = law->J_p;
				ex0 = law->full.edge_x[0];
				ex1 = law->full.edge_x[1];
				r_patch.mSpinRadius = float(law->r_s);
				r_result.m1[0] = law->m1x;
				r_result.m1[1] = law->m1y;
			}
			r_patch.mArea = float(area);

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
				r_patch.mTiltArmPos[i] = float(face_v2 ? f2_edge_distance(*law2, mx, my) : edge_distance(law->full, mx, my));
				r_patch.mTiltArmNeg[i] = float(face_v2 ? f2_edge_distance(*law2, -mx, -my) : edge_distance(law->full, -mx, -my));
			}

			// The tail's contact sets for the next step's first pass
			if (face_v2) {
				const F2Grid &g2 = law2->grid;
				for (int kind = 0; kind < 2; ++kind) {
					const F2Field &f = law2->field[kind];
					if (!f.solved) {
						continue;
					}
					for (int k = 0; k < g2.n; ++k) {
						if (f.active[k]) {
							r_patch.mActiveSet[kind * F2_TAIL_WORDS + (k >> 5)] |= 1u << (k & 31);
						}
					}
				}
				r_patch.mGridLong = uint8_t(g2.nx);
				r_patch.mGridShort = uint8_t(g2.ny);
				r_patch.mGridKind = law2->field[0].solved ? JPH::SoftContactTail::GridFaceV2 : JPH::SoftContactTail::GridFaceV2NoLub;
				// and the held slide's memory after this step
				r_patch.mHeldGradient[0] = float(law2->held_out[0]);
				r_patch.mHeldGradient[1] = float(law2->held_out[1]);
				r_patch.mHeldArea = float(law2->held_out[2]);
			} else {
				const Grid &g = law->full.grid;
				for (int k = 0; k < g.n() && k < 192; ++k) {
					if (law->full.active[k]) {
						r_patch.mActiveSet[k >> 5] |= 1u << (k & 31);
					}
				}
				r_patch.mGridLong = uint8_t(g.nl);
				r_patch.mGridShort = uint8_t(g.ns);
				r_patch.mGridKind = JPH::SoftContactTail::GridSplit;
			}

			// Pre-slip shear (Layer): G A / h per tangent, G J_p / h about the normal
			r_patch.mShearStiffness = float(co.G * area / soft.thickness);
			r_patch.mSpinStiffness = float(co.G * J_p / soft.thickness);
			r_patch.mRelaxationLength = float(0.5 * (ex1 - ex0));
			r_patch.mNormalDampingRatio = float(loss_side.zeta_for(1.0));
		}

		if (soft.kind == JoltSoftSurface3D::KIND_LAYER) {
			// The law for a first touch and across a gap: the normal stiffness of the footprints in full contact. Only a
			// gap or an undecided first touch needs it; a pressed patch whose first touch is decided keeps its own.
			double k_ref = kn;
			if (!pressed || onset_pending) {
				if (face_v2) {
					thread_local F2Law scratch_ref_v2;
					F2Law *ref2 = &scratch_ref_v2;
					if (f2_face_law(soft_poly, partners, 1.0, 0.0, 0.0, soft.modulus, soft.bulk_modulus, soft.thickness, mu, F2_EDGE_TOL, nullptr, *ref2) && !ref2->empty) {
						k_ref = ref2->k_n;
					}
				} else {
					thread_local FaceLaw scratch_ref;
					FaceLaw *ref = &scratch_ref;
					if (face_law(soft_poly, partners, 1.0 * share, 0.0, 0.0, co, nullptr, *ref) && !ref->empty) {
						k_ref = ref->k_n * (lubricated_series ? share : table_ratio(false, shape_factor(ref->area, ref->Ixx, ref->Iyy, soft.thickness), mu, kg_of(soft)));
					}
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
	if (face_edge || (cls == 1 && !(R > 0.0))) {
		// a polyhedral edge: rounded to the layer's depth (design 4.1 item 2); a face's edge is one whatever the leaf
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
	} else if (face_edge) {
		// A face's vertex: the overlap is its depth past the flat face along n, at the midpoint between the two
		body_tangents(body2, n, t1, t2);
		const HeightPlane at = face_plane(*flat, curved->v[0], t1, t2, n);
		if (!at.valid) {
			r_result.outcome = JoltSoftBuildResult::UNCLASSED;
			return false;
		}
		s0 = curved_is_1 ? -at.z0 : at.z0;
		origin = curved->v[0] + float(0.5 * at.z0) * n;
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
		const Coeffs co = split_coeffs(soft.modulus, soft.bulk_modulus, soft.thickness, false);
		shear_G = co.G;
		// Two curved soft sides (Stage 1c-b fix round, C3): the lead's decision 5 through the table, each side's layer with
		// a lubricated face, the compliances in series (two_sided_split); where a side is outside the table, the side with
		// the larger local share's own law at that share (two_sided_outside, counted as outside the table)
		const JoltSoftSurface3D *lead_side = two_layers ? (two_sided_share0(*s1, *s2) >= 0.5 ? s1 : s2) : &soft;
		if (s0 > 0.0 || (cls == 1 && d_deep > 0.0)) {
			if (two_layers) {
				have = cls == 1 ? table_line(s0, camber, ya, yb, R, *s1, 0.0, fig, s2) : two_sided_point(s0, R, *s1, *s2, fig);
				if (!have) {
					have = two_sided_outside(cls == 1, s0, R, *s1, *s2, camber, ya, yb, fig);
					r_result.outside_table = have && !fig.empty;
				}
			} else {
				have = cls == 1 ? table_line(s0, camber, ya, yb, R, soft, mu, fig) : table_point(s0, R, soft, mu, fig);
			}
			if (!have && !two_layers) {
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
		// The law from first touch: the table at a reference squash of a thousandth of the depth (two soft sides: of the
		// layers' whole depth, through the series)
		CurvedFig ref;
		const double s_ref = 1e-3 * (two_layers ? layer : soft.thickness);
		bool ref_ok = two_layers ? (cls == 1 ? table_line(s_ref, 0.0, ya, yb, R, *s1, 0.0, ref, s2) : two_sided_point(s_ref, R, *s1, *s2, ref)) : (cls == 1 ? table_line(s_ref, 0.0, ya, yb, R, soft, mu, ref) : table_point(s_ref, R, soft, mu, ref));
		if (ref_ok && !ref.empty && ref.F > 0.0) {
			const double nexp = MAX(ref.n_exp, 1.0);
			r_patch.mLawExponent = float(nexp);
			r_patch.mLawCoefficient = float(ref.F / std::pow(s_ref, nexp));
		} else if (have && !fig.empty && s0 > 0.0) {
			const double nexp = MAX(fig.n_exp, 1.0);
			r_patch.mLawExponent = float(nexp);
			r_patch.mLawCoefficient = float(fig.F / std::pow(MAX(d_deep, 1e-9), nexp));
		} else if (two_layers) {
			// Two soft sides outside the table across a gap: the law the pressed patch uses there (two_sided_outside), read
			// at the squash where the leading side's contact half-width reaches its depth, sqrt(2 R s) = h. The grid cannot
			// resolve a contact much narrower than its cells (N_SHORT cells across a box of half-width sqrt(2 R s) + h): at
			// a thousandth of the depth it finds no contact at all.
			const double hl = lead_side->thickness;
			const double s_two = MAX(1e-3 * layer, hl * hl / (2.0 * R));
			CurvedFig ref2;
			if (two_sided_outside(cls == 1, s_two, R, *s1, *s2, 0.0, ya, yb, ref2) && !ref2.empty && ref2.F > 0.0) {
				const double nexp = MAX(ref2.n_exp, 1.0);
				r_patch.mLawExponent = float(nexp);
				r_patch.mLawCoefficient = float(ref2.F / std::pow(s_two, nexp));
			}
		}
		// The first-touch law the solver reads (Stage 1c-b): the pressed law itself at six squashes, the same law this
		// patch uses when it presses. In the table, from a thousandth of the depth to the table's last squash (0.316 of
		// it), the level line's slices or the point's figures (two curved soft sides: of the layers' whole depth, through
		// the series); outside the table, or where one lookup fails, the pressed fallback (the split law on the grid with
		// its face ratio; two sides: two_sided_outside) from where the grid resolves the contact (a thousandth of the
		// depth, or a quarter of the squash at which the contact's half-width reaches the depth, h^2 / 8R) to half the
		// layers' whole depth.
		const bool pressing = have && !fig.empty && fig.F > 0.0;
		if (onset_pending || !pressing) {
			int kept = 0;
			if (two_layers) {
				kept = set_law_samples(1e-3 * layer, 0.316 * layer, 6, true, [&](double p_s, double &r_F) {
					CurvedFig f;
					const bool ok = cls == 1 ? table_line(p_s, 0.0, ya, yb, R, *s1, 0.0, f, s2) : two_sided_point(p_s, R, *s1, *s2, f);
					r_F = f.F;
					return ok && !f.empty;
				}, r_patch);
			} else {
				kept = set_law_samples(1e-3 * soft.thickness, 0.316 * soft.thickness, 6, true, [&](double p_s, double &r_F) {
					CurvedFig f;
					const bool ok = cls == 1 ? table_line(p_s, 0.0, ya, yb, R, soft, mu, f) : table_point(p_s, R, soft, mu, f);
					r_F = f.F;
					return ok && !f.empty;
				}, r_patch);
			}
			if (kept == 0 && two_layers) {
				const double hh = lead_side->thickness;
				set_law_samples(MAX(1e-3 * hh, 0.25 * hh * hh / (2.0 * R)), 0.5 * layer, 6, false, [&](double p_s, double &r_F) {
					CurvedFig f;
					if (!two_sided_outside(cls == 1, p_s, R, *s1, *s2, 0.0, ya, yb, f) || f.empty || !(f.F > 0.0)) {
						return false;
					}
					r_F = f.F;
					return true;
				}, r_patch);
			} else if (kept == 0) {
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
		if (!rigid_stick && two_layers) {
			// Two soft sides (Stage 1c-b fix round, C3): the two layers' shear in series, A / (h1/G1 + h2/G2) per tangent
			// and J_p / (h1/G1 + h2/G2) in spin
			double G1, nu1, G2, nu2;
			elastic(s1->modulus, s1->bulk_modulus, G1, nu1);
			elastic(s2->modulus, s2->bulk_modulus, G2, nu2);
			const double c = s1->thickness / G1 + s2->thickness / G2;
			r_patch.mShearStiffness = float(fig.area / c);
			r_patch.mSpinStiffness = float(fig.J_p / c);
		} else if (!rigid_stick) {
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
