/*
 * Turbo work schedule: which (z, y) rows each layer computes. Shared by
 * metal_turbo.h and webgpu_turbo.h; plain C, no GPU API.
 *
 * BC_TURBO_CONE=1 (default) clones brainchop-test V4's cone skip
 * (computeConeSchedule in webgpu_runners/meshnet_fast_v4_runner.js), per layer
 * whichever is cheaper:
 *  - PURE (forward, exact): a row whose every tap reads background rows inside
 *    the volume equals every other such row voxel for voxel, x edges included
 *    (all share the same x profile and padding). One representative is
 *    computed, weighted by the count in the moments, and copied to the rest.
 *  - CONE (backward, approximate): only rows within 3 of a non-background
 *    input row are labelled; the rest are labelled 0 without being computed.
 *    The last layers compute only what the next one reads. GroupNorm moments
 *    of the skipped exterior are estimated from a stratified lattice of rows,
 *    computed with their support but not otherwise used.
 * The cone engages only when the input's first and last voxels share the
 * background value and some, but not almost all, rows differ from it.
 * BC_TURBO_CONE=0 computes every row of every layer.
 */
#ifndef BRAINCHOPC_TURBO_CONE_H
#define BRAINCHOPC_TURBO_CONE_H

#include "model_specs.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef BC_TURBO_CONE
#define BC_TURBO_CONE 1
#endif

enum { BC_TC_ROWS = 65536 };

typedef struct {
	uint32_t list_off, count; /* computed rows and their moment weights */
	uint32_t pure_off, npure; /* rows copied from `rep` after the norm */
	uint32_t rep;
} bc_tc_layer;

typedef struct {
	bc_tc_layer layer[BC_MODEL_NHIDDEN];
	uint32_t cls_off, ncls;   /* rows the classifier labels; the rest stay 0 */
	uint32_t *list;           /* row indices, all ranges above */
	float *weight;            /* moment weight per list entry */
	size_t n;
} bc_tc_plan;

/* Safe to call twice: bc_tc_make frees on failure and callers free again. */
static void bc_tc_free(bc_tc_plan *plan) {
	free(plan->list);
	free(plan->weight);
	plan->list = NULL;
	plan->weight = NULL;
}

static void bc_tc_put(bc_tc_plan *p, uint32_t row, float w) {
	p->list[p->n] = row;
	p->weight[p->n++] = w;
}

#if BC_TURBO_CONE
enum { BC_TC_FULL, BC_TC_PURE, BC_TC_CONE };

/* Mark in `out` the rows that computing the rows of `in` at dilation d reads. */
static void bc_tc_spread(const uint8_t *in, uint8_t *out, int d) {
	memset(out, 0, BC_TC_ROWS);
	for (int z = 0; z < 256; z++) for (int y = 0; y < 256; y++) {
		if (!in[z * 256 + y]) continue;
		for (int dz = -d; dz <= d; dz += d) for (int dy = -d; dy <= d; dy += d) {
			int zz = z + dz, yy = y + dy;
			if (zz >= 0 && zz < 256 && yy >= 0 && yy < 256) out[zz * 256 + yy] = 1;
		}
	}
}

static int bc_tc_count(const uint8_t *m) {
	int n = 0;
	for (int r = 0; r < BC_TC_ROWS; r++) n += m[r];
	return n;
}
#endif

/* Returns 0, or 1 when out of memory.
   allow_cone=1 enables both exact pure-background copies (PURE) and the
   approximate backward-cone lattice (CONE); allow_cone=0 uses exact PURE only
   (0.000 Dice/voxel drift from full-volume moments). */
static int bc_tc_make_ex(const unsigned char *conformed, const int *dilation,
		bc_tc_plan *p, int allow_cone) {
	enum { L = BC_MODEL_NHIDDEN, N = BC_TC_ROWS };
	(void)conformed; (void)dilation; (void)allow_cone;
	memset(p, 0, sizeof *p);
	/* Every layer lists N rows at most (a pure layer's computed and copied rows
	   partition them), plus the classifier rows. */
	size_t cap = (size_t)(L + 1) * N;
	p->list = (uint32_t *)malloc(cap * sizeof *p->list);
	p->weight = (float *)malloc(cap * sizeof *p->weight);
	if (!p->list || !p->weight) { bc_tc_free(p); return 1; }
#if BC_TURBO_CONE
	int mode[L];
	for (int l = 0; l < L; l++) mode[l] = BC_TC_FULL;
	uint8_t *m = (uint8_t *)calloc((size_t)(3 * L + 4) * N, 1);
	if (!m) { bc_tc_free(p); return 1; }
	uint8_t *head = m, *cls = m + N, *lat = m + 2 * N, *tmp = m + 3 * N;
	uint8_t *pure = m + 4 * N, *cone = pure + (size_t)L * N, *comp = cone + (size_t)L * N;
	const unsigned char bg = conformed[0];
	int nhead = 0;
	if (conformed[(size_t)N * 256 - 1] == bg) {
		for (int r = 0; r < N; r++) {
			const unsigned char *v = conformed + (size_t)r * 256;
			int any = 0;
			for (int x = 0; x < 256; x++) any |= v[x] != bg;
			head[r] = (uint8_t)any;
			nhead += any;
		}
	}
	/* V4's bounds: nothing to skip, or nothing worth skipping. */
	if (nhead > 0 && nhead < 60000) {
		int work[L];
		/* Forward: pure[l]. Rows within d of a z/y face read zero padding. */
		for (int r = 0; r < N; r++) tmp[r] = !head[r];
		const uint8_t *prev = tmp;
		for (int l = 0; l < L; l++) {
			const int d = dilation[l];
			uint8_t *cur = pure + (size_t)l * N;
			for (int z = d; z < 256 - d; z++) for (int y = d; y < 256 - d; y++) {
				int ok = 1;
				for (int dz = -d; ok && dz <= d; dz += d) for (int dy = -d; ok && dy <= d; dy += d)
					ok = prev[(z + dz) * 256 + y + dy];
				cur[z * 256 + y] = (uint8_t)ok;
			}
			const int npure = bc_tc_count(cur);
			work[l] = npure >= N / 100 ? N - npure + 1 : N;
			mode[l] = work[l] < N ? BC_TC_PURE : BC_TC_FULL;
			prev = cur;
		}
		if (allow_cone) {
			/* Backward: cone = cone of the labelled rows; comp adds the lattice
			   and the support the next layer's computed rows read. */
			for (int z = 0; z < 256; z++) for (int y = 0; y < 256; y++) {
				int any = 0;
				for (int zz = z - 3; !any && zz <= z + 3; zz++) for (int yy = y - 3; !any && yy <= y + 3; yy++)
					any = zz >= 0 && zz < 256 && yy >= 0 && yy < 256 && head[zz * 256 + yy];
				cls[z * 256 + y] = (uint8_t)any;
			}
			/* 1024 rows, one per 8x8 cell, jittered (V4: a Korobov lattice of
			   256-512 four-row strips). */
			for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++)
				lat[(8 * i + ((3 * i + 5 * j) & 7)) * 256 + 8 * j + ((5 * i + 3 * j) & 7)] = 1;
			/* V4's thresholds: cone only if it skips >3.6% of labels, per layer
			   only if it computes <15/16 of rows. */
			if (bc_tc_count(cls) < N * 964 / 1000) {
				for (int l = L - 1; l >= 0; l--) {
					uint8_t *ml = cone + (size_t)l * N, *cl = comp + (size_t)l * N;
					if (l == L - 1) {
						memcpy(ml, cls, N);
						memset(tmp, 0, N);
					} else {
						bc_tc_spread(cone + (size_t)(l + 1) * N, ml, dilation[l + 1]);
						bc_tc_spread(comp + (size_t)(l + 1) * N, tmp, dilation[l + 1]);
					}
					int nlat = 0, ncomp = 0;
					for (int r = 0; r < N; r++) {
						cl[r] = ml[r] ? 1 : lat[r] ? 2 : tmp[r] ? 3 : 0;
						nlat += cl[r] == 2;
						ncomp += cl[r] != 0;
					}
					if (nlat && ncomp <= N * 15 / 16 && ncomp < work[l]) mode[l] = BC_TC_CONE;
					if (ncomp > N * 15 / 16) break; /* earlier layers only grow */
				}
				/* A cone layer computes only what the next layer reads, so it must
				   be followed by one. */
				for (int l = L - 2; l >= 0; l--)
					if (mode[l] == BC_TC_CONE && mode[l + 1] != BC_TC_CONE) mode[l] = BC_TC_FULL;
			}
		}
	}
#endif
	for (int l = 0; l < L; l++) {
		bc_tc_layer *t = &p->layer[l];
		t->list_off = (uint32_t)p->n;
#if BC_TURBO_CONE
		if (mode[l] == BC_TC_PURE) {
			const uint8_t *cur = pure + (size_t)l * N;
			int rep = 0;
			while (!cur[rep]) rep++;
			t->rep = (uint32_t)rep;
			for (int r = 0; r < N; r++) if (!cur[r]) bc_tc_put(p, (uint32_t)r, 1.0f);
			bc_tc_put(p, (uint32_t)rep, (float)(N - (int)(p->n - t->list_off)));
			t->count = (uint32_t)p->n - t->list_off;
			t->pure_off = (uint32_t)p->n;
			for (int r = 0; r < N; r++) if (cur[r] && r != rep) bc_tc_put(p, (uint32_t)r, 0.0f);
			t->npure = (uint32_t)p->n - t->pure_off;
			continue;
		}
		if (mode[l] == BC_TC_CONE) {
			const uint8_t *cl = comp + (size_t)l * N;
			int nmain = 0, nlat = 0;
			for (int r = 0; r < N; r++) { nmain += cl[r] == 1; nlat += cl[r] == 2; }
			const float w[4] = { 0.0f, 1.0f, (float)(N - nmain) / (float)nlat, 0.0f };
			for (int r = 0; r < N; r++) if (cl[r]) bc_tc_put(p, (uint32_t)r, w[cl[r]]);
			t->count = (uint32_t)p->n - t->list_off;
			continue;
		}
#endif
		for (int r = 0; r < N; r++) bc_tc_put(p, (uint32_t)r, 1.0f);
		t->count = (uint32_t)p->n - t->list_off;
	}
	/* Labels: the cone's labelled rows, else every computed row (a pure last
	   layer's copies are labelled by copying the representative's labels). */
	p->cls_off = p->layer[L - 1].list_off;
	p->ncls = p->layer[L - 1].count;
#if BC_TURBO_CONE
	if (mode[L - 1] == BC_TC_CONE) {
		p->cls_off = (uint32_t)p->n;
		for (int r = 0; r < N; r++) if (cls[r]) bc_tc_put(p, (uint32_t)r, 0.0f);
		p->ncls = (uint32_t)p->n - p->cls_off;
	}
	free(m);
#endif
	return 0;
}

static inline int bc_tc_make(const unsigned char *conformed, const int *dilation, bc_tc_plan *p) {
	return bc_tc_make_ex(conformed, dilation, p, 1);
}

#endif
