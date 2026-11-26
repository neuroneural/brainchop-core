#include "bwlabel.h"
#ifdef BC_THREADS
#include "bc_threads.h"
#endif
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef BC_THREADS
static void bw_parallel_for(int lo, int hi, int grain, bc_parallel_fn fn, void *ctx) {
	bc_parallel_for(lo, hi, grain, fn, ctx);
}
#else
static void bw_parallel_for(int lo, int hi, int grain,
		void (*fn)(void *, int, int), void *ctx) {
	(void)grain;
	if (lo <= hi) fn(ctx, lo, hi);
}
#endif

/* ========================================================================== *
 * Fast parallel z-slab scanline union-find helpers                           *
 * ========================================================================== */

#define BW_MAX_SLABS 16

typedef struct {
	uint32_t *parent;
	uint32_t *size;
	uint32_t *first;
	uint8_t *klass;
	uint32_t count;
	uint32_t cap;
	int failed;
} bw_dsu;

static void bw_dsu_free(bw_dsu *d) {
	free(d->parent);
	free(d->size);
	free(d->first);
	free(d->klass);
	memset(d, 0, sizeof(*d));
}

static int bw_dsu_reserve(bw_dsu *d, uint32_t need, int with_klass) {
	if (need <= d->cap) return 0;
	uint32_t ncap = d->cap ? d->cap * 2u : 4096u;
	while (ncap < need) {
		if (ncap > UINT32_MAX / 2u) { ncap = need; break; }
		ncap *= 2u;
	}
	uint32_t *np = realloc(d->parent, (size_t)ncap * sizeof(*np));
	if (!np) return 1;
	d->parent = np;
	uint32_t *ns = realloc(d->size, (size_t)ncap * sizeof(*ns));
	if (!ns) return 1;
	d->size = ns;
	uint32_t *nf = realloc(d->first, (size_t)ncap * sizeof(*nf));
	if (!nf) return 1;
	d->first = nf;
	if (with_klass) {
		uint8_t *nk = realloc(d->klass, (size_t)ncap * sizeof(*nk));
		if (!nk) return 1;
		d->klass = nk;
	}
	if (d->cap == 0) {
		d->parent[0] = 0;
		d->size[0] = 0;
		d->first[0] = UINT32_MAX;
		if (with_klass) d->klass[0] = 0;
	}
	d->cap = ncap;
	return 0;
}

static inline uint32_t dsu_find(uint32_t *parent, uint32_t x) {
	uint32_t r = x;
	while (parent[r] != r) r = parent[r];
	while (parent[x] != r) {
		uint32_t next = parent[x];
		parent[x] = r;
		x = next;
	}
	return r;
}

static inline uint32_t dsu_union(uint32_t *parent, uint32_t *size,
		uint32_t *first, uint32_t a, uint32_t b) {
	a = dsu_find(parent, a);
	b = dsu_find(parent, b);
	if (a == b) return a;
	if (a > b) { uint32_t t = a; a = b; b = t; }
	parent[b] = a;
	size[a] += size[b];
	if (first[b] < first[a]) first[a] = first[b];
	return a;
}

static inline uint32_t dsu_union_plain(uint32_t *parent, uint32_t a, uint32_t b) {
	a = dsu_find(parent, a);
	b = dsu_find(parent, b);
	if (a == b) return a;
	if (a > b) { uint32_t t = a; a = b; b = t; }
	parent[b] = a;
	return a;
}

static int choose_slabs(size_t nz) {
#ifdef BC_THREADS
	if (nz < 32) return 1;
	int s = (int)(nz / 16);
	if (s > BW_MAX_SLABS) s = BW_MAX_SLABS;
	if (s < 1) s = 1;
	return s;
#else
	(void)nz;
	return 1;
#endif
}

/* ========================================================================== *
 * bwlabel_largest: 6- or 26-connected largest nonzero component              *
 * ========================================================================== */

typedef struct {
	const float *img;
	uint8_t *fg;
	size_t xy;
} bw_fg_extract_ctx;

static void bw_fg_extract_rows(void *ctx, int z0, int z1) {
	const bw_fg_extract_ctx *c = ctx;
	size_t start = (size_t)z0 * c->xy;
	size_t end = (size_t)(z1 + 1) * c->xy;
	const float *img = c->img;
	uint8_t *fg = c->fg;
	for (size_t i = start; i < end; i++)
		fg[i] = img[i] > 0.0f ? 1u : 0u;
}

typedef struct {
	const uint8_t *fg;
	uint32_t *map;
	size_t nx, ny, nz, xy;
	int connectivity;
	int nslabs;
	int slab_z0[BW_MAX_SLABS];
	int slab_z1[BW_MAX_SLABS];
	bw_dsu dsu[BW_MAX_SLABS];
} bw_bin_slab_ctx;

static void bw_bin_slab_run(void *ctx, int s0, int s1) {
	bw_bin_slab_ctx *c = ctx;
	const size_t nx = c->nx, ny = c->ny, xy = c->xy;
	const uint8_t *fg = c->fg;
	uint32_t *map = c->map;
	const int conn26 = c->connectivity == 26;

	for (int s = s0; s <= s1; s++) {
		bw_dsu *d = &c->dsu[s];
		const int z0 = c->slab_z0[s], z1 = c->slab_z1[s];
		for (int z = z0; z <= z1; z++) {
			const size_t z_off = (size_t)z * xy;
			for (size_t y = 0; y < ny; y++) {
				const size_t row_off = z_off + y * nx;
				const uint8_t *row = fg + row_off;
				uint32_t *mrow = map + row_off;
				size_t x = 0;
				while (x < nx) {
					/* Fast skip zeros using 8-byte chunks when aligned */
					while (x + 8 <= nx && !row[x] && !row[x+1] && !row[x+2] && !row[x+3] &&
							!row[x+4] && !row[x+5] && !row[x+6] && !row[x+7])
						x += 8;
					while (x < nx && !row[x]) x++;
					if (x >= nx) break;
					size_t rx0 = x;
					while (x < nx && row[x]) x++;
					size_t rx1 = x;
					uint32_t run_len = (uint32_t)(rx1 - rx0);
					uint32_t root = 0;

					if (conn26) {
						size_t nb0 = rx0 > 0 ? rx0 - 1 : 0;
						size_t nb1 = rx1 < nx ? rx1 + 1 : nx;
						/* Previous row in same slice: (z, y-1) */
						if (y > 0) {
							const uint32_t *pm = mrow - nx;
							for (size_t k = nb0; k < nb1; k++) {
								uint32_t l = pm[k];
								if (l) root = root ? dsu_union(d->parent, d->size, d->first, root, l) : dsu_find(d->parent, l);
							}
						}
						/* Previous slice inside this slab: (z-1, y-1..y+1) */
						if (z > z0) {
							size_t py0 = y > 0 ? y - 1 : 0;
							size_t py1 = y + 1 < ny ? y + 1 : ny - 1;
							const uint32_t *prev_sl = map + (size_t)(z - 1) * xy;
							for (size_t py = py0; py <= py1; py++) {
								const uint32_t *pm = prev_sl + py * nx;
								for (size_t k = nb0; k < nb1; k++) {
									uint32_t l = pm[k];
									if (l) root = root ? dsu_union(d->parent, d->size, d->first, root, l) : dsu_find(d->parent, l);
								}
							}
						}
					} else {
						/* 6-connected: only (z, y-1, rx0..rx1) and (z-1, y, rx0..rx1) */
						if (y > 0) {
							const uint32_t *pm = mrow - nx;
							for (size_t k = rx0; k < rx1; k++) {
								uint32_t l = pm[k];
								if (l) root = root ? dsu_union(d->parent, d->size, d->first, root, l) : dsu_find(d->parent, l);
							}
						}
						if (z > z0) {
							const uint32_t *pm = mrow - xy;
							for (size_t k = rx0; k < rx1; k++) {
								uint32_t l = pm[k];
								if (l) root = root ? dsu_union(d->parent, d->size, d->first, root, l) : dsu_find(d->parent, l);
							}
						}
					}

					uint32_t first_idx = (uint32_t)(row_off + rx0);
					if (!root) {
						uint32_t id = ++d->count;
						if (bw_dsu_reserve(d, id + 1u, 0)) { d->failed = 1; return; }
						d->parent[id] = id;
						d->size[id] = run_len;
						d->first[id] = first_idx;
						root = id;
					} else {
						d->size[root] += run_len;
						if (first_idx < d->first[root]) d->first[root] = first_idx;
					}
					for (size_t k = rx0; k < rx1; k++) mrow[k] = root;
				}
			}
		}
	}
}

typedef struct {
	float *img;
	const uint32_t *map;
	const uint8_t *keep;
	size_t xy;
	int binary_output;
	int slab_z0[BW_MAX_SLABS];
	int slab_z1[BW_MAX_SLABS];
	uint32_t slab_off[BW_MAX_SLABS];
} bw_largest_apply_ctx;

static void bw_largest_apply_slab(void *ctx, int s0, int s1) {
	const bw_largest_apply_ctx *c = ctx;
	for (int s = s0; s <= s1; s++) {
		size_t start = (size_t)c->slab_z0[s] * c->xy;
		size_t end = (size_t)(c->slab_z1[s] + 1) * c->xy;
		uint32_t off = c->slab_off[s];
		float *img = c->img;
		const uint32_t *map = c->map;
		const uint8_t *keep = c->keep;
		if (c->binary_output) {
			for (size_t i = start; i < end; i++) {
				uint32_t l = map[i];
				if (!l || !keep[off + l]) img[i] = 0.0f;
				else img[i] = 1.0f;
			}
		} else {
			for (size_t i = start; i < end; i++) {
				uint32_t l = map[i];
				if (!l || !keep[off + l]) img[i] = 0.0f;
			}
		}
	}
}

/*
 * Keep the largest nonzero component of a 3-D label image.
 * Tie-break preserves raster-order discovery (first component wins ties).
 */
int bwlabel_largest(float *img, const size_t dim[3], int binary_output,
		int connectivity) {
	if (!img || !dim || dim[0] < 1 || dim[1] < 1 || dim[2] < 1)
		return 1;
	if (connectivity != 6 && connectivity != 26) return 1;
	if (dim[0] > SIZE_MAX / dim[1] || dim[0] * dim[1] > SIZE_MAX / dim[2])
		return 1;
	const size_t nx = dim[0], ny = dim[1], nz = dim[2];
	const size_t xy = nx * ny, n = xy * nz;
	if (n > UINT32_MAX || n > SIZE_MAX / sizeof(uint32_t)) return 1;

	uint8_t *fg = malloc(n);
	uint32_t *map = calloc(n, sizeof(*map));
	if (!fg || !map) {
		fprintf(stderr, "bwlabel: out of memory\n");
		free(fg); free(map);
		return 1;
	}

	bw_fg_extract_ctx ext = { .img = img, .fg = fg, .xy = xy };
	bw_parallel_for(0, (int)nz - 1, 8, bw_fg_extract_rows, &ext);

	int nslabs = choose_slabs(nz);
	bw_bin_slab_ctx sc;
	memset(&sc, 0, sizeof(sc));
	sc.fg = fg; sc.map = map;
	sc.nx = nx; sc.ny = ny; sc.nz = nz; sc.xy = xy;
	sc.connectivity = connectivity;
	sc.nslabs = nslabs;
	for (int s = 0; s < nslabs; s++) {
		sc.slab_z0[s] = (int)((size_t)s * nz / (size_t)nslabs);
		sc.slab_z1[s] = (int)((size_t)(s + 1) * nz / (size_t)nslabs) - 1;
	}
	bw_parallel_for(0, nslabs - 1, 1, bw_bin_slab_run, &sc);

	int failed = 0;
	uint32_t slab_off[BW_MAX_SLABS];
	size_t total_prov = 0;
	for (int s = 0; s < nslabs; s++) {
		if (sc.dsu[s].failed) failed = 1;
		slab_off[s] = (uint32_t)total_prov;
		total_prov += sc.dsu[s].count;
	}
	if (failed || total_prov > UINT32_MAX - 1u) {
		for (int s = 0; s < nslabs; s++) bw_dsu_free(&sc.dsu[s]);
		free(fg); free(map);
		fprintf(stderr, "bwlabel: out of memory\n");
		return 1;
	}
	if (total_prov == 0) {
		for (int s = 0; s < nslabs; s++) bw_dsu_free(&sc.dsu[s]);
		free(fg); free(map);
		memset(img, 0, n * sizeof(*img));
		return 0;
	}

	uint32_t *gparent = malloc((total_prov + 1) * sizeof(*gparent));
	uint32_t *gsize = malloc((total_prov + 1) * sizeof(*gsize));
	uint32_t *gfirst = malloc((total_prov + 1) * sizeof(*gfirst));
	uint8_t *keep = calloc(total_prov + 1, 1);
	if (!gparent || !gsize || !gfirst || !keep) {
		for (int s = 0; s < nslabs; s++) bw_dsu_free(&sc.dsu[s]);
		free(gparent); free(gsize); free(gfirst); free(keep);
		free(fg); free(map);
		fprintf(stderr, "bwlabel: out of memory\n");
		return 1;
	}
	gparent[0] = 0; gsize[0] = 0; gfirst[0] = UINT32_MAX;
	for (int s = 0; s < nslabs; s++) {
		uint32_t off = slab_off[s];
		const bw_dsu *d = &sc.dsu[s];
		for (uint32_t i = 1; i <= d->count; i++) {
			gparent[off + i] = off + d->parent[i];
			gsize[off + i] = d->size[i];
			gfirst[off + i] = d->first[i];
		}
		bw_dsu_free(&sc.dsu[s]);
	}

	/* Merge touching foreground voxels across the nslabs - 1 z-seams. */
	const int conn26 = connectivity == 26;
	for (int s = 0; s + 1 < nslabs; s++) {
		uint32_t off_a = slab_off[s], off_b = slab_off[s + 1];
		const uint32_t *sl_a = map + (size_t)sc.slab_z1[s] * xy;
		const uint32_t *sl_b = map + (size_t)sc.slab_z0[s + 1] * xy;
		for (size_t y = 0; y < ny; y++) {
			const uint32_t *ra = sl_a + y * nx;
			if (!conn26) {
				const uint32_t *rb = sl_b + y * nx;
				for (size_t x = 0; x < nx; x++) {
					if (ra[x] && rb[x])
						dsu_union(gparent, gsize, gfirst, off_a + ra[x], off_b + rb[x]);
				}
			} else {
				size_t py0 = y > 0 ? y - 1 : 0;
				size_t py1 = y + 1 < ny ? y + 1 : ny - 1;
				for (size_t x = 0; x < nx; x++) {
					uint32_t la = ra[x];
					if (!la) continue;
					uint32_t ga = off_a + la;
					size_t px0 = x > 0 ? x - 1 : 0;
					size_t px1 = x + 1 < nx ? x + 1 : nx - 1;
					for (size_t py = py0; py <= py1; py++) {
						const uint32_t *rb = sl_b + py * nx;
						for (size_t px = px0; px <= px1; px++) {
							uint32_t lb = rb[px];
							if (lb) dsu_union(gparent, gsize, gfirst, ga, off_b + lb);
						}
					}
				}
			}
		}
	}

	uint32_t best_root = 0, best_count = 0, best_first = UINT32_MAX;
	for (uint32_t i = 1; i <= (uint32_t)total_prov; i++) {
		if (gparent[i] != i) continue;
		if (gsize[i] > best_count || (gsize[i] == best_count && gfirst[i] < best_first)) {
			best_count = gsize[i];
			best_first = gfirst[i];
			best_root = i;
		}
	}
	for (uint32_t i = 1; i <= (uint32_t)total_prov; i++) {
		if (dsu_find(gparent, i) == best_root) keep[i] = 1;
	}

	bw_largest_apply_ctx app;
	app.img = img; app.map = map; app.keep = keep;
	app.xy = xy; app.binary_output = binary_output;
	for (int s = 0; s < nslabs; s++) {
		app.slab_z0[s] = sc.slab_z0[s];
		app.slab_z1[s] = sc.slab_z1[s];
		app.slab_off[s] = slab_off[s];
	}
	bw_parallel_for(0, nslabs - 1, 1, bw_largest_apply_slab, &app);

	free(gparent); free(gsize); free(gfirst); free(keep);
	free(fg); free(map);
	return 0;
}

/* ========================================================================== *
 * bwlabel_validate: parallel validity, fill-ratio, and fragmentation guard   *
 * ========================================================================== */

typedef struct {
	const float *img;
	size_t nx, ny, nz, xy;
	unsigned classes;
	size_t fg[256];
	size_t roots[256];
	size_t bad_idx[256];
	int has_bad[256];
	int nchunks;
} bw_validate_ctx;

static void bw_validate_chunk(void *ctx, int c0, int c1) {
	bw_validate_ctx *c = ctx;
	const float *img = c->img;
	const size_t nx = c->nx, ny = c->ny, nz = c->nz, xy = c->xy;
	const double cls = (double)c->classes;
	for (int ch = c0; ch <= c1; ch++) {
		size_t z0 = (size_t)ch * nz / (size_t)c->nchunks;
		size_t z1 = (size_t)(ch + 1) * nz / (size_t)c->nchunks;
		size_t fg = 0, roots = 0;
		for (size_t z = z0; z < z1; z++) {
			size_t zoff = z * xy;
			for (size_t y = 0; y < ny; y++) {
				size_t yoff = zoff + y * nx;
				for (size_t x = 0; x < nx; x++) {
					size_t i = yoff + x;
					float v = img[i];
					if (!isfinite(v) || v < 0.0f || (double)v >= cls || v != floorf(v)) {
						c->has_bad[ch] = 1;
						c->bad_idx[ch] = i;
						return;
					}
					if (v == 0.0f) continue;
					fg++;
					if ((x == 0 || img[i - 1] != v) &&
							(y == 0 || img[i - nx] != v) &&
							(z == 0 || img[i - xy] != v))
						roots++;
				}
			}
		}
		c->fg[ch] = fg;
		c->roots[ch] = roots;
	}
}

static inline int face_neighbors_fast(uint32_t p, size_t nx, size_t ny,
		size_t nz, size_t xy, int is256, uint32_t q[6]) {
	size_t x, y, z;
	if (is256) {
		x = p & 255u;
		y = (p >> 8) & 255u;
		z = p >> 16;
	} else {
		z = p / xy;
		size_t rem = p - z * xy;
		y = rem / nx;
		x = rem - y * nx;
	}
	int n = 0;
	if (x) q[n++] = p - 1u;
	if (x + 1 < nx) q[n++] = p + 1u;
	if (y) q[n++] = p - (uint32_t)nx;
	if (y + 1 < ny) q[n++] = p + (uint32_t)nx;
	if (z) q[n++] = p - (uint32_t)xy;
	if (z + 1 < nz) q[n++] = p + (uint32_t)xy;
	return n;
}

int bwlabel_validate(const float *img, const size_t dim[3], unsigned classes) {
	if (!img || !dim || !classes || !dim[0] || !dim[1] || !dim[2] ||
			dim[0] > SIZE_MAX / dim[1] || dim[0] * dim[1] > SIZE_MAX / dim[2]) {
		fprintf(stderr, "brainchopc: invalid categorical volume dimensions\n");
		return 1;
	}
	const size_t nx = dim[0], ny = dim[1], nz = dim[2];
	const size_t xy = nx * ny, n = xy * nz;
	if (n > UINT32_MAX || n > SIZE_MAX / sizeof(uint32_t)) return 1;

	bw_validate_ctx vc;
	memset(&vc, 0, sizeof(vc));
	vc.img = img; vc.nx = nx; vc.ny = ny; vc.nz = nz; vc.xy = xy;
	vc.classes = classes;
	vc.nchunks = nz >= 64 ? 64 : (int)nz;
	bw_parallel_for(0, vc.nchunks - 1, 1, bw_validate_chunk, &vc);

	size_t foreground = 0, roots = 0;
	for (int ch = 0; ch < vc.nchunks; ch++) {
		if (vc.has_bad[ch]) {
			size_t i = vc.bad_idx[ch];
			fprintf(stderr, "brainchopc: invalid categorical label at voxel %zu: %g\n",
				i, (double)img[i]);
			return 1;
		}
		foreground += vc.fg[ch];
		roots += vc.roots[ch];
	}
	if (!foreground || (double)foreground > (double)n * 0.95) {
		fprintf(stderr, "brainchopc: categorical foreground is %zu/%zu voxels; %s\n",
			foreground, n, foreground ? "segmentation is near-filled" : "segmentation is empty");
		return 1;
	}
	size_t cap = n / 100;
	if (cap < 100000) cap = 100000;
	if (roots <= cap) return 0;

	/* Exact fallback BFS when upper bound exceeds cap (noisy/corrupted output). */
	unsigned char *visited = calloc(n, 1);
	uint32_t *queue = malloc(foreground * sizeof(*queue));
	if (!visited || !queue) {
		free(visited); free(queue);
		fprintf(stderr, "brainchopc: unable to allocate categorical quality workspace\n");
		return 1;
	}
	const int is256 = (nx == 256 && ny == 256);
	size_t components = 0;
	for (size_t i = 0; i < n; i++) {
		if (!img[i] || visited[i]) continue;
		if (++components > cap) break;
		size_t head = 0, tail = 1;
		queue[0] = (uint32_t)i; visited[i] = 1;
		float seed_v = img[i];
		while (head < tail) {
			uint32_t q[6];
			int nq = face_neighbors_fast(queue[head++], nx, ny, nz, xy, is256, q);
			for (int j = 0; j < nq; j++)
				if (!visited[q[j]] && img[q[j]] == seed_v) {
					visited[q[j]] = 1;
					queue[tail++] = q[j];
				}
		}
	}
	free(visited); free(queue);
	if (components > cap)
		fprintf(stderr, "brainchopc: categorical segmentation is excessively fragmented "
			"(more than %zu six-connected regions)\n", cap);
	return components > cap;
}

/* ========================================================================== *
 * Hierarchical 6-connected cleanup (18-class & 104-class)                    *
 * Single parallel z-slab per-class UF + component-graph binary brain LCC     *
 * ========================================================================== */

static int targeted18(unsigned klass) {
	return klass == 1 || klass == 2 || klass == 5 || klass == 6 || klass == 13;
}

static int targeted_all(unsigned klass) {
	return klass > 0;
}

static unsigned dk104_cortical_family(unsigned klass) {
	if (klass >= 1 && klass <= 34) return 1;
	if (klass >= 35 && klass <= 68) return 2;
	return 0;
}

typedef struct {
	const float *img;
	uint8_t *cls;
	size_t xy;
	unsigned nclass;
	int invalid;
} bw_cls_extract_ctx;

static void bw_cls_extract_rows(void *ctx, int z0, int z1) {
	bw_cls_extract_ctx *c = ctx;
	size_t start = (size_t)z0 * c->xy;
	size_t end = (size_t)(z1 + 1) * c->xy;
	const float *img = c->img;
	uint8_t *cls = c->cls;
	const double nc = (double)c->nclass;
	for (size_t i = start; i < end; i++) {
		float v = img[i];
		if (!isfinite(v) || v < 0.0f || (double)v >= nc || v != (float)(unsigned)v) {
			c->invalid = 1;
			return;
		}
		cls[i] = (uint8_t)(unsigned)v;
	}
}

typedef struct {
	const uint8_t *cls;
	uint32_t *map;
	size_t nx, ny, nz, xy;
	int nslabs;
	int slab_z0[BW_MAX_SLABS];
	int slab_z1[BW_MAX_SLABS];
	bw_dsu dsu[BW_MAX_SLABS];
} bw_cls_slab_ctx;

static void bw_cls_slab_run(void *ctx, int s0, int s1) {
	bw_cls_slab_ctx *c = ctx;
	const size_t nx = c->nx, ny = c->ny, xy = c->xy;
	const uint8_t *cls = c->cls;
	uint32_t *map = c->map;

	for (int s = s0; s <= s1; s++) {
		bw_dsu *d = &c->dsu[s];
		const int z0 = c->slab_z0[s], z1 = c->slab_z1[s];
		for (int z = z0; z <= z1; z++) {
			const size_t z_off = (size_t)z * xy;
			for (size_t y = 0; y < ny; y++) {
				const size_t row_off = z_off + y * nx;
				const uint8_t *row = cls + row_off;
				uint32_t *mrow = map + row_off;
				size_t x = 0;
				while (x < nx) {
					while (x + 8 <= nx && !row[x] && !row[x+1] && !row[x+2] && !row[x+3] &&
							!row[x+4] && !row[x+5] && !row[x+6] && !row[x+7])
						x += 8;
					while (x < nx && !row[x]) x++;
					if (x >= nx) break;
					uint8_t val = row[x];
					size_t rx0 = x;
					while (x < nx && row[x] == val) x++;
					size_t rx1 = x;
					uint32_t run_len = (uint32_t)(rx1 - rx0);
					uint32_t root = 0;

					if (y > 0) {
						const uint8_t *prow = row - nx;
						const uint32_t *pm = mrow - nx;
						for (size_t k = rx0; k < rx1; k++) {
							if (prow[k] == val) {
								uint32_t l = pm[k];
								root = root ? dsu_union(d->parent, d->size, d->first, root, l) : dsu_find(d->parent, l);
							}
						}
					}
					if (z > z0) {
						const uint8_t *prow = row - xy;
						const uint32_t *pm = mrow - xy;
						for (size_t k = rx0; k < rx1; k++) {
							if (prow[k] == val) {
								uint32_t l = pm[k];
								root = root ? dsu_union(d->parent, d->size, d->first, root, l) : dsu_find(d->parent, l);
							}
						}
					}

					uint32_t first_idx = (uint32_t)(row_off + rx0);
					if (!root) {
						uint32_t id = ++d->count;
						if (bw_dsu_reserve(d, id + 1u, 1)) { d->failed = 1; return; }
						d->parent[id] = id;
						d->size[id] = run_len;
						d->first[id] = first_idx;
						d->klass[id] = val;
						root = id;
					} else {
						d->size[root] += run_len;
						if (first_idx < d->first[root]) d->first[root] = first_idx;
					}
					for (size_t k = rx0; k < rx1; k++) mrow[k] = root;
				}
			}
		}
	}
}

typedef struct {
	uint32_t *map;
	const uint32_t *dense_id;
	size_t xy;
	int slab_z0[BW_MAX_SLABS];
	int slab_z1[BW_MAX_SLABS];
	uint32_t slab_off[BW_MAX_SLABS];
} bw_remap_ctx;

static void bw_remap_slab(void *ctx, int s0, int s1) {
	const bw_remap_ctx *c = ctx;
	for (int s = s0; s <= s1; s++) {
		size_t start = (size_t)c->slab_z0[s] * c->xy;
		size_t end = (size_t)(c->slab_z1[s] + 1) * c->xy;
		uint32_t off = c->slab_off[s];
		uint32_t *map = c->map;
		const uint32_t *dense = c->dense_id;
		for (size_t i = start; i < end; i++) {
			uint32_t l = map[i];
			if (l) map[i] = dense[off + l];
		}
	}
}

typedef struct {
	const uint8_t *cls;
	const uint32_t *map;
	size_t nx, ny, nz, xy;
	uint32_t count;
	int slab_z0[BW_MAX_SLABS];
	int slab_z1[BW_MAX_SLABS];
	uint32_t *bin_parent[BW_MAX_SLABS];
} bw_bin_adj_ctx;

static void bw_bin_adj_slab(void *ctx, int s0, int s1) {
	const bw_bin_adj_ctx *c = ctx;
	const size_t nx = c->nx, ny = c->ny, xy = c->xy;
	const uint8_t *cls = c->cls;
	const uint32_t *map = c->map;
	for (int s = s0; s <= s1; s++) {
		uint32_t *bp = c->bin_parent[s];
		for (int z = c->slab_z0[s]; z <= c->slab_z1[s]; z++) {
			const size_t z_off = (size_t)z * xy;
			for (size_t y = 0; y < ny; y++) {
				const size_t row_off = z_off + y * nx;
				const uint8_t *row = cls + row_off;
				const uint32_t *mrow = map + row_off;
				size_t x = 0;
				while (x < nx) {
					while (x + 8 <= nx && !row[x] && !row[x+1] && !row[x+2] && !row[x+3] &&
							!row[x+4] && !row[x+5] && !row[x+6] && !row[x+7])
						x += 8;
					while (x < nx && !row[x]) x++;
					if (x >= nx) break;
					uint8_t val = row[x];
					uint32_t cid = mrow[x];
					if (x > 0 && row[x - 1]) {
						uint32_t nb = mrow[x - 1];
						if (nb != cid) dsu_union_plain(bp, cid, nb);
					}
					size_t rx0 = x;
					while (x < nx && row[x] == val) x++;
					size_t rx1 = x;
					if (y > 0) {
						const uint32_t *pm = mrow - nx;
						for (size_t k = rx0; k < rx1; k++) {
							uint32_t nb = pm[k];
							if (nb && nb != cid) dsu_union_plain(bp, cid, nb);
						}
					}
					if (z > 0) {
						const uint32_t *pm = mrow - xy;
						for (size_t k = rx0; k < rx1; k++) {
							uint32_t nb = pm[k];
							if (nb && nb != cid) dsu_union_plain(bp, cid, nb);
						}
					}
				}
			}
		}
	}
}

typedef struct {
	float *img;
	const uint32_t *map;
	const uint8_t *replacement;
	size_t xy;
} bw_writeback_ctx;

static void bw_writeback_rows(void *ctx, int z0, int z1) {
	const bw_writeback_ctx *c = ctx;
	size_t start = (size_t)z0 * c->xy;
	size_t end = (size_t)(z1 + 1) * c->xy;
	float *img = c->img;
	const uint32_t *map = c->map;
	const uint8_t *rep = c->replacement;
	for (size_t i = start; i < end; i++)
		img[i] = (float)rep[map[i]];
}

static int bwlabel_cleanup_hierarchical(float *img, const size_t dim[3],
		unsigned nclass, int (*is_targeted)(unsigned), int larger_index_wins,
		unsigned (*family_of)(unsigned)) {
	if (!img || !dim || !nclass || nclass > 256 || !dim[0] || !dim[1] || !dim[2] ||
			dim[0] > SIZE_MAX / dim[1] || dim[0] * dim[1] > SIZE_MAX / dim[2])
		return 1;
	const size_t nx = dim[0], ny = dim[1], nz = dim[2];
	const size_t xy = nx * ny, n = xy * nz;
	if (n >= UINT32_C(0x80000000) || n > SIZE_MAX / sizeof(uint32_t)) return 1;

	uint8_t *cls = malloc(n);
	uint32_t *map = calloc(n, sizeof(*map));
	if (!cls || !map) {
		free(cls); free(map);
		fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
		return 1;
	}

	/* 1. Validate floats and pack into 1-byte labels in parallel. */
	bw_cls_extract_ctx ext = { .img = img, .cls = cls, .xy = xy, .nclass = nclass, .invalid = 0 };
	bw_parallel_for(0, (int)nz - 1, 8, bw_cls_extract_rows, &ext);
	if (ext.invalid) {
		free(cls); free(map);
		return 1;
	}

	/* 2. Single parallel z-slab 6-connected per-class scanline union-find. */
	int nslabs = choose_slabs(nz);
	bw_cls_slab_ctx sc;
	memset(&sc, 0, sizeof(sc));
	sc.cls = cls; sc.map = map;
	sc.nx = nx; sc.ny = ny; sc.nz = nz; sc.xy = xy;
	sc.nslabs = nslabs;
	for (int s = 0; s < nslabs; s++) {
		sc.slab_z0[s] = (int)((size_t)s * nz / (size_t)nslabs);
		sc.slab_z1[s] = (int)((size_t)(s + 1) * nz / (size_t)nslabs) - 1;
	}
	bw_parallel_for(0, nslabs - 1, 1, bw_cls_slab_run, &sc);

	int failed = 0;
	uint32_t slab_off[BW_MAX_SLABS];
	size_t total_prov = 0;
	for (int s = 0; s < nslabs; s++) {
		if (sc.dsu[s].failed) failed = 1;
		slab_off[s] = (uint32_t)total_prov;
		total_prov += sc.dsu[s].count;
	}
	if (failed || total_prov > UINT32_MAX - 1u) {
		for (int s = 0; s < nslabs; s++) bw_dsu_free(&sc.dsu[s]);
		free(cls); free(map);
		fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
		return 1;
	}
	if (total_prov == 0) {
		for (int s = 0; s < nslabs; s++) bw_dsu_free(&sc.dsu[s]);
		free(cls); free(map);
		return 0;
	}

	uint32_t *gparent = malloc((total_prov + 1) * sizeof(*gparent));
	uint32_t *gsize = malloc((total_prov + 1) * sizeof(*gsize));
	uint32_t *gfirst = malloc((total_prov + 1) * sizeof(*gfirst));
	uint8_t *gklass = malloc((total_prov + 1) * sizeof(*gklass));
	uint32_t *dense_id = calloc(total_prov + 1, sizeof(*dense_id));
	if (!gparent || !gsize || !gfirst || !gklass || !dense_id) {
		for (int s = 0; s < nslabs; s++) bw_dsu_free(&sc.dsu[s]);
		free(gparent); free(gsize); free(gfirst); free(gklass); free(dense_id);
		free(cls); free(map);
		fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
		return 1;
	}
	gparent[0] = 0; gsize[0] = 0; gfirst[0] = UINT32_MAX; gklass[0] = 0;
	for (int s = 0; s < nslabs; s++) {
		uint32_t off = slab_off[s];
		const bw_dsu *d = &sc.dsu[s];
		for (uint32_t i = 1; i <= d->count; i++) {
			gparent[off + i] = off + d->parent[i];
			gsize[off + i] = d->size[i];
			gfirst[off + i] = d->first[i];
			gklass[off + i] = d->klass[i];
		}
		bw_dsu_free(&sc.dsu[s]);
	}

	/* Merge same-class components across the nslabs - 1 z-seams. */
	for (int s = 0; s + 1 < nslabs; s++) {
		uint32_t off_a = slab_off[s], off_b = slab_off[s + 1];
		const uint8_t *ca = cls + (size_t)sc.slab_z1[s] * xy;
		const uint8_t *cb = cls + (size_t)sc.slab_z0[s + 1] * xy;
		const uint32_t *ma = map + (size_t)sc.slab_z1[s] * xy;
		const uint32_t *mb = map + (size_t)sc.slab_z0[s + 1] * xy;
		for (size_t i = 0; i < xy; i++) {
			if (ca[i] && ca[i] == cb[i])
				dsu_union(gparent, gsize, gfirst, off_a + ma[i], off_b + mb[i]);
		}
	}

	/* Assign dense final component IDs 1..count. */
	uint32_t count = 0;
	for (uint32_t i = 1; i <= (uint32_t)total_prov; i++) {
		if (gparent[i] == i) dense_id[i] = ++count;
	}
	for (uint32_t i = 1; i <= (uint32_t)total_prov; i++) {
		dense_id[i] = dense_id[dsu_find(gparent, i)];
	}

	uint32_t *comp_size = calloc((size_t)count + 1, sizeof(*comp_size));
	uint32_t *comp_first = malloc(((size_t)count + 1) * sizeof(*comp_first));
	uint8_t *comp_klass = calloc((size_t)count + 1, 1);
	uint8_t *comp_kept = calloc((size_t)count + 1, 1);
	uint8_t *comp_rep = calloc((size_t)count + 1, 1);
	uint32_t *bin_parent = malloc(((size_t)count + 1) * sizeof(*bin_parent));
	if (!comp_size || !comp_first || !comp_klass || !comp_kept || !comp_rep || !bin_parent) {
		free(gparent); free(gsize); free(gfirst); free(gklass); free(dense_id);
		free(comp_size); free(comp_first); free(comp_klass);
		free(comp_kept); free(comp_rep); free(bin_parent);
		free(cls); free(map);
		fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
		return 1;
	}
	comp_first[0] = UINT32_MAX;
	bin_parent[0] = 0;
	for (uint32_t i = 1; i <= (uint32_t)total_prov; i++) {
		if (gparent[i] != i) continue;
		uint32_t c = dense_id[i];
		comp_size[c] = gsize[i];
		comp_first[c] = gfirst[i];
		comp_klass[c] = gklass[i];
		bin_parent[c] = c;
	}
	free(gparent); free(gsize); free(gfirst); free(gklass);

	/* Remap map[] in parallel to dense per-class component IDs 1..count. */
	bw_remap_ctx rmc;
	rmc.map = map; rmc.dense_id = dense_id; rmc.xy = xy;
	for (int s = 0; s < nslabs; s++) {
		rmc.slab_z0[s] = sc.slab_z0[s];
		rmc.slab_z1[s] = sc.slab_z1[s];
		rmc.slab_off[s] = slab_off[s];
	}
	bw_parallel_for(0, nslabs - 1, 1, bw_remap_slab, &rmc);
	free(dense_id);

	/* 3. Binary 6-connected brain LCC over the per-class component graph. */
	bw_bin_adj_ctx bac;
	memset(&bac, 0, sizeof(bac));
	bac.cls = cls; bac.map = map;
	bac.nx = nx; bac.ny = ny; bac.nz = nz; bac.xy = xy;
	bac.count = count;
	int adj_slabs = ((size_t)count * (size_t)nslabs * sizeof(uint32_t) <= (16u << 20)) ? nslabs : 1;
	int adj_oom = 0;
	for (int s = 0; s < adj_slabs; s++) {
		bac.slab_z0[s] = (int)((size_t)s * nz / (size_t)adj_slabs);
		bac.slab_z1[s] = (int)((size_t)(s + 1) * nz / (size_t)adj_slabs) - 1;
		if (s == 0) {
			bac.bin_parent[0] = bin_parent;
		} else {
			bac.bin_parent[s] = malloc(((size_t)count + 1) * sizeof(uint32_t));
			if (!bac.bin_parent[s]) { adj_oom = 1; break; }
			memcpy(bac.bin_parent[s], bin_parent, ((size_t)count + 1) * sizeof(uint32_t));
		}
	}
	if (adj_oom) {
		for (int s = 1; s < adj_slabs; s++) free(bac.bin_parent[s]);
		bac.slab_z0[0] = 0; bac.slab_z1[0] = (int)nz - 1;
		adj_slabs = 1;
	}
	bw_parallel_for(0, adj_slabs - 1, 1, bw_bin_adj_slab, &bac);
	for (int s = 1; s < adj_slabs; s++) {
		uint32_t *bp = bac.bin_parent[s];
		for (uint32_t c = 1; c <= count; c++) {
			if (bp[c] != c) dsu_union_plain(bin_parent, c, bp[c]);
		}
		free(bp);
	}

	uint32_t *bin_size = calloc((size_t)count + 1, sizeof(*bin_size));
	uint32_t *bin_first = malloc(((size_t)count + 1) * sizeof(*bin_first));
	if (!bin_size || !bin_first) {
		free(bin_size); free(bin_first);
		free(comp_size); free(comp_first); free(comp_klass);
		free(comp_kept); free(comp_rep); free(bin_parent);
		free(cls); free(map);
		fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
		return 1;
	}
	for (uint32_t c = 0; c <= count; c++) bin_first[c] = UINT32_MAX;
	for (uint32_t c = 1; c <= count; c++) {
		uint32_t r = dsu_find(bin_parent, c);
		bin_parent[c] = r;
		bin_size[r] += comp_size[c];
		if (comp_first[c] < bin_first[r]) bin_first[r] = comp_first[c];
	}
	/* Browser binary LCC keeps the LAST tie (largest raster seed index). */
	uint32_t best_bin = 0, best_bin_size = 0, best_bin_first = 0;
	for (uint32_t c = 1; c <= count; c++) {
		if (bin_parent[c] != c) continue;
		if (bin_size[c] > best_bin_size ||
				(bin_size[c] == best_bin_size && bin_first[c] >= best_bin_first)) {
			best_bin_size = bin_size[c];
			best_bin_first = bin_first[c];
			best_bin = c;
		}
	}
	free(bin_size); free(bin_first);

	/* 4. Per-class largest component inside best_bin. */
	uint32_t largest[256] = {0};
	for (uint32_t c = 1; c <= count; c++) {
		if (bin_parent[c] != best_bin) continue;
		unsigned k = comp_klass[c];
		uint32_t prev = largest[k];
		if (!prev) {
			largest[k] = c;
		} else {
			uint32_t sz = comp_size[c], psz = comp_size[prev];
			/* 18-class keeps FIRST tie (smallest first_voxel); 104-class keeps LAST. */
			if (sz > psz || (sz == psz && (larger_index_wins
					? comp_first[c] > comp_first[prev]
					: comp_first[c] < comp_first[prev])))
				largest[k] = c;
		}
	}
	uint32_t n_suppressed = 0;
	uint32_t *sup_slot = NULL;
	for (uint32_t c = 1; c <= count; c++) {
		if (bin_parent[c] != best_bin) continue;
		unsigned k = comp_klass[c];
		if (!is_targeted(k) || largest[k] == c) {
			comp_kept[c] = comp_rep[c] = (uint8_t)k;
		} else {
			n_suppressed++;
		}
	}

	/* 5. Boundary voting for suppressed components inside best_bin. */
	if (n_suppressed > 0) {
		size_t vote_elems = (size_t)n_suppressed * (size_t)nclass;
		if (vote_elems <= (4u << 20)) {
			/* Fast single-scan face-vote accumulation (no BFS needed). */
			sup_slot = calloc((size_t)count + 1, sizeof(*sup_slot));
			uint32_t *votes = calloc(vote_elems, sizeof(*votes));
			if (!sup_slot || !votes) {
				free(sup_slot); free(votes);
				free(comp_size); free(comp_first); free(comp_klass);
				free(comp_kept); free(comp_rep); free(bin_parent);
				free(cls); free(map);
				fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
				return 1;
			}
			uint32_t slot = 0;
			for (uint32_t c = 1; c <= count; c++) {
				if (bin_parent[c] == best_bin && !comp_kept[c])
					sup_slot[c] = ++slot; /* 1-based */
			}
			for (size_t z = 0; z < nz; z++) {
				const size_t z_off = z * xy;
				for (size_t y = 0; y < ny; y++) {
					const size_t row_off = z_off + y * nx;
					const uint8_t *row = cls + row_off;
					const uint32_t *mrow = map + row_off;
					size_t x = 0;
					while (x < nx) {
						while (x + 8 <= nx && !row[x] && !row[x+1] && !row[x+2] && !row[x+3] &&
								!row[x+4] && !row[x+5] && !row[x+6] && !row[x+7])
							x += 8;
						while (x < nx && !row[x]) x++;
						if (x >= nx) break;
						uint32_t cid = mrow[x];
						uint32_t sl = sup_slot[cid];
						if (!sl) {
							uint8_t v = row[x];
							while (x < nx && row[x] == v) x++;
							continue;
						}
						uint32_t *vrow = votes + (size_t)(sl - 1u) * nclass;
						uint8_t v = row[x];
						for (; x < nx && row[x] == v; x++) {
							uint8_t k;
							if (x > 0 && (k = comp_kept[mrow[x - 1]])) vrow[k]++;
							if (x + 1 < nx && (k = comp_kept[mrow[x + 1]])) vrow[k]++;
							if (y > 0 && (k = comp_kept[(mrow - nx)[x]])) vrow[k]++;
							if (y + 1 < ny && (k = comp_kept[(mrow + nx)[x]])) vrow[k]++;
							if (z > 0 && (k = comp_kept[(mrow - xy)[x]])) vrow[k]++;
							if (z + 1 < nz && (k = comp_kept[(mrow + xy)[x]])) vrow[k]++;
						}
					}
				}
			}
			for (uint32_t c = 1; c <= count; c++) {
				uint32_t sl = sup_slot[c];
				if (!sl) continue;
				const uint32_t *vrow = votes + (size_t)(sl - 1u) * nclass;
				unsigned winner = 0;
				uint32_t best_votes = 0;
				unsigned fam = family_of ? family_of(comp_klass[c]) : 0;
				if (fam) {
					for (unsigned k = 1; k < nclass; k++)
						if (family_of(k) == fam && vrow[k] > best_votes) {
							best_votes = vrow[k];
							winner = k;
						}
				}
				if (!winner) {
					best_votes = 0;
					for (unsigned k = 1; k < nclass; k++)
						if (vrow[k] > best_votes) {
							best_votes = vrow[k];
							winner = k;
						}
				}
				comp_rep[c] = (uint8_t)winner;
			}
			free(sup_slot);
			free(votes);
		} else {
			/* Fallback BFS for pathological inputs with >40K suppressed components. */
			uint32_t *queue = malloc(n * sizeof(*queue));
			if (!queue) {
				free(comp_size); free(comp_first); free(comp_klass);
				free(comp_kept); free(comp_rep); free(bin_parent);
				free(cls); free(map);
				fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
				return 1;
			}
			const uint32_t vis_bit = UINT32_C(0x80000000);
			const int is256 = (nx == 256 && ny == 256);
			for (uint32_t c = 1; c <= count; c++) {
				if (bin_parent[c] != best_bin || comp_kept[c]) continue;
				uint64_t votes[256] = {0};
				uint32_t head = 0, tail = 1;
				queue[0] = comp_first[c];
				map[queue[0]] |= vis_bit;
				while (head < tail) {
					uint32_t q[6];
					int nq = face_neighbors_fast(queue[head++], nx, ny, nz, xy, is256, q);
					for (int j = 0; j < nq; j++) {
						uint32_t id = map[q[j]];
						if (id == c) {
							map[q[j]] |= vis_bit;
							queue[tail++] = q[j];
						} else {
							unsigned k = comp_kept[id & ~vis_bit];
							if (k) votes[k]++;
						}
					}
				}
				for (uint32_t q = 0; q < tail; q++) map[queue[q]] &= ~vis_bit;
				unsigned winner = 0;
				uint64_t best_votes = 0;
				unsigned fam = family_of ? family_of(comp_klass[c]) : 0;
				if (fam) {
					for (unsigned k = 1; k < nclass; k++)
						if (family_of(k) == fam && votes[k] > best_votes) {
							best_votes = votes[k];
							winner = k;
						}
				}
				if (!winner) {
					best_votes = 0;
					for (unsigned k = 1; k < nclass; k++)
						if (votes[k] > best_votes) {
							best_votes = votes[k];
							winner = k;
						}
				}
				comp_rep[c] = (uint8_t)winner;
			}
			free(queue);
		}
	}

	/* 6. Parallel float write-back. */
	bw_writeback_ctx wbc = { .img = img, .map = map, .replacement = comp_rep, .xy = xy };
	bw_parallel_for(0, (int)nz - 1, 8, bw_writeback_rows, &wbc);

	free(comp_size); free(comp_first); free(comp_klass);
	free(comp_kept); free(comp_rep); free(bin_parent);
	free(cls); free(map);
	return 0;
}

int bwlabel_cleanup18(float *img, const size_t dim[3]) {
	return bwlabel_cleanup_hierarchical(img, dim, 18, targeted18, 0, NULL);
}

int bwlabel_cleanup104(float *img, const size_t dim[3]) {
	return bwlabel_cleanup_hierarchical(img, dim, 104, targeted_all, 1,
		dk104_cortical_family);
}
