/*
 * clock_gettime is POSIX, not ISO C, and this file is built with -std=c11 which
 * hides it. The host stage profiler needs a monotonic clock. Must precede every
 * include -- note the comment below celebrating the removal of an earlier
 * _POSIX_C_SOURCE preamble; this one is back for a different reason.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
/*
 * The weights are compiled into this executable (see payload.S), so there is no
 * longer any need to locate the executable on disk. That removed the only uses
 * of readlink(), realpath() and _NSGetExecutablePath(), and with them the
 * _POSIX_C_SOURCE preamble those prototypes required on Linux.
 */

#include "brainchopc.h"
#include "backend.h"
#include "bwlabel.h"
#include "backend_accel.h"
#include "cat_lite.h"
#include "cli.h"
#include "model_specs.h"
#include "payload.h"
#include "preprocess.h"
#include "version.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bc_threads.h"

/*
 * Host stage timing, gated on BRAINCHOPC_PROFILE.
 *
 * The GPU backends each carry a stage profiler (BRAINCHOPC_CUDA_PROFILE and
 * friends) and the host side carried none, which was fine while the host side
 * was noise: the project's percentages were measured against a 140 s CPU run,
 * where conform was ~0.1% and gzip "within noise".
 *
 * That stopped being true. A tuned CUDA run is ~1.08 s of which only ~0.35 s is
 * on the GPU, so roughly two thirds of it is host work that nothing measured --
 * and it barely responds to the thread count (1.115 s at one thread, 1.085 s at
 * 32 -- measured through OMP_NUM_THREADS, which is inert now that threading is
 * pthreads; the spellings are `-cpu N` and BC_NUM_THREADS), so most of it is
 * serial. This exists to find out which part.
 */
static double bc_now_ms(void) {
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

static int bc_profiling(void) {
	static int state = -1;
	if (state < 0) state = getenv("BRAINCHOPC_PROFILE") ? 1 : 0;
	return state;
}

/*
 * NOT a partition of wall time. Two windows are deliberately unmarked -- process
 * start to the first mark, and the DT_FLOAT32 convert plus sanitize_input
 * between the read and run_model -- so the stages sum to less than the clock.
 * Read them as "where the big pieces are", not as an accounting.
 */
static double bc_stage_start = 0.0;
static void bc_stage(const char *label) {
	if (!bc_profiling()) return;
	double now = bc_now_ms();
	if (bc_stage_start > 0.0 && label)
		fprintf(stderr, "brainchopc: host %-18s %8.1f ms\n", label,
			now - bc_stage_start);
	bc_stage_start = now;
}

int nii_mul_size(size_t a, size_t b, size_t *out) {
	if (!out || (a && b > SIZE_MAX / a)) return 1;
	*out = a * b;
	return 0;
}

void *nii_malloc(size_t count, size_t size) {
	size_t bytes;
	if (nii_mul_size(count, size, &bytes)) return NULL;
	void *p = malloc(bytes);
	if (!p && bytes) fprintf(stderr, "brainchopc: out of memory\n");
	return p;
}

vec4 setVec4(float x, float y, float z) {
	vec4 v = {{x,y,z,1}};
	return v;
}

vec4 nifti_vect44mat44_mul(vec4 v, mat44 m) {
	vec4 out = {{0,0,0,0}};
	for (int r=0;r<4;r++) for(int c=0;c<4;c++) out.v[r] += m.m[r][c]*v.v[c];
	return out;
}

static double voxel(const nifti_image *nim, size_t i) {
	switch (nim->datatype) {
	case DT_UINT8: return ((uint8_t *)nim->data)[i];
	case DT_INT8: return ((int8_t *)nim->data)[i];
	case DT_UINT16: return ((uint16_t *)nim->data)[i];
	case DT_INT16: return ((int16_t *)nim->data)[i];
	case DT_UINT32: return ((uint32_t *)nim->data)[i];
	case DT_INT32: return ((int32_t *)nim->data)[i];
	case DT_UINT64: return (double)((uint64_t *)nim->data)[i];
	case DT_INT64: return (double)((int64_t *)nim->data)[i];
	case DT_FLOAT32: return ((float *)nim->data)[i];
	case DT_FLOAT64: return ((double *)nim->data)[i];
	default: return NAN;
	}
}

static double clamp_round(double x, double lo, double hi) {
	if (x < lo) return lo;
	if (x > hi) return hi;
	return round(x);
}

/*
 * The 64-bit integer cases cannot go through clamp_round, and the reason is a
 * trap rather than a style preference: its bounds are doubles, and both
 * (double)INT64_MAX and (double)UINT64_MAX round UP to the first out-of-range
 * power of two. A value "clamped" to one of those still converts out of range,
 * which is undefined behaviour and not a wrap. 2^63 and 2^64 ARE exactly
 * representable, so the bound to test is the power of two, exclusive above.
 *
 * Reachable, not theoretical: voxel() reads both datatypes, so a 64-bit input
 * round-trips through float32 -- where INT64_MAX becomes exactly 2^63 -- and
 * comes back here to be restored.
 *
 * Negated comparisons, so a NaN takes a clamp branch instead of the cast. That
 * is belt-and-braces rather than a live hole: the only call with an integer
 * target is the restore at the end of a run, by which point the earlier
 * conversion to float has already forced scl_slope to 1 and scl_inter to 0 and
 * sanitize_input has replaced every non-finite voxel. MAGNITUDE is what makes
 * this reachable, not NaN -- and note clamp_round above shares the NaN
 * assumption for the same reason.
 *
 * round() cannot push a value back out of range: every double at or above 2^52
 * is already an integer, so it is the identity everywhere near these bounds.
 */
static uint64_t round_u64(double x) {
	if (!(x > 0.0)) return 0;
	if (!(x < 18446744073709551616.0)) return UINT64_MAX;   /* 2^64 */
	return (uint64_t)round(x);
}

static int64_t round_i64(double x) {
	if (!(x >= -9223372036854775808.0)) return INT64_MIN;   /* -2^63 */
	if (!(x < 9223372036854775808.0)) return INT64_MAX;     /* 2^63 */
	return (int64_t)round(x);
}

/*
 * Every stage downstream of here converts float to int somewhere -- histogram
 * bins in conform, nearest-neighbour indices in reslice, output quantisation --
 * and those conversions are undefined for NaN. Rather than guard each one, make
 * the data finite once, at the single point where a user file enters.
 *
 * voxel() decodes unreadable datatypes (complex, RGB) as NaN, so the same pass
 * also catches them; an all-NaN volume otherwise ran to completion and wrote a
 * full-size image of +Inf with exit status 0.
 */
static int sanitize_input(nifti_image *nim, int source_datatype) {
	float *value = nim->data;
	size_t finite = 0, replaced = 0;
	for (size_t i = 0; i < (size_t)nim->nvox; i++) {
		if (isfinite(value[i])) { finite++; continue; }
		value[i] = 0.0f;
		replaced++;
	}
	if (!finite) {
		/* Report the datatype the file actually had: the caller has already
		   converted nim->datatype to float by this point. */
		fprintf(stderr, "brainchopc: input has no finite voxels "
			"(datatype %d is unreadable, or the image is entirely NaN)\n",
			source_datatype);
		return 1;
	}
	if (replaced)
		fprintf(stderr, "brainchopc: WARNING - replaced %zu non-finite voxels with 0\n",
			replaced);
	return 0;
}

int brainchopc_convert(nifti_image *nim, int datatype) {
	if (!nim || !nim->data || nim->nvox < 1) return 1;
	if (nim->datatype == datatype && nim->scl_slope == 1.0 && nim->scl_inter == 0.0) return 0;
	int nbyper=0;
	nifti_datatype_sizes(datatype,&nbyper,NULL);
	if (nbyper < 1 || datatype == DT_RGB24 || datatype == DT_RGBA32) return 1;
	void *out=nii_malloc((size_t)nim->nvox,(size_t)nbyper);
	if (!out) return 1;
	double slope=(!isfinite(nim->scl_slope) || nim->scl_slope == 0.0) ? 1.0 : nim->scl_slope;
	double inter=isfinite(nim->scl_inter) ? nim->scl_inter : 0.0;
	for (size_t i=0;i<(size_t)nim->nvox;i++) {
		double v=voxel(nim,i)*slope+inter;
		switch(datatype) {
		case DT_UINT8: ((uint8_t*)out)[i]=(uint8_t)clamp_round(v,0,UINT8_MAX); break;
		case DT_INT8: ((int8_t*)out)[i]=(int8_t)clamp_round(v,INT8_MIN,INT8_MAX); break;
		case DT_UINT16: ((uint16_t*)out)[i]=(uint16_t)clamp_round(v,0,UINT16_MAX); break;
		case DT_INT16: ((int16_t*)out)[i]=(int16_t)clamp_round(v,INT16_MIN,INT16_MAX); break;
		case DT_UINT32: ((uint32_t*)out)[i]=(uint32_t)clamp_round(v,0,UINT32_MAX); break;
		case DT_INT32: ((int32_t*)out)[i]=(int32_t)clamp_round(v,INT32_MIN,INT32_MAX); break;
		case DT_UINT64: ((uint64_t*)out)[i]=round_u64(v); break;
		case DT_INT64: ((int64_t*)out)[i]=round_i64(v); break;
		case DT_FLOAT32: ((float*)out)[i]=(float)v; break;
		case DT_FLOAT64: ((double*)out)[i]=v; break;
		default: free(out); return 1;
		}
	}
	free(nim->data);
	nim->data=out; nim->datatype=datatype; nim->nbyper=nbyper;
	nim->scl_slope=1; nim->scl_inter=0;
	return 0;
}

static nifti_image *private_float_copy(const nifti_image *src) {
	nifti_image *dst=calloc(1,sizeof(*dst));
	if (!dst) return NULL;
	*dst=*src;
	dst->fname=NULL; dst->iname=NULL; dst->ext_list=NULL; dst->num_ext=0;
	size_t bytes;
	if (nii_mul_size((size_t)src->nvox,sizeof(float),&bytes)) { free(dst); return NULL; }
	dst->data=malloc(bytes);
	if (!dst->data) { free(dst); return NULL; }
	memcpy(dst->data,src->data,bytes);
	return dst;
}

/*
 * Locate this build's embedded weights for a model. The size is checked against
 * what the generated descriptor says the model contains: the Makefile makes the
 * same assertion before assembling, so reaching a mismatch here means the
 * payload and the descriptor were built from different models -- exactly the
 * drift `make check-generated` exists to prevent, caught once more at the point
 * of use because the consequence is a plausible-looking wrong segmentation
 * rather than a crash.
 */
static const float *model_payload(size_t *count) {
	size_t bytes = (size_t)(bc_model_payload_end - bc_model_payload_begin);
	if (bytes != (size_t)BC_MODEL_WEIGHT_FLOATS * sizeof(float)) {
		fprintf(stderr,"brainchop: embedded weights are %zu bytes but this "
			"build expects %zu\n", bytes,
			(size_t)BC_MODEL_WEIGHT_FLOATS * sizeof(float));
		return NULL;
	}
	*count = BC_MODEL_WEIGHT_FLOATS;
	return (const float *)(const void *)bc_model_payload_begin;
}

static float *load_model_weights(void) {
	size_t count=0;
	const float *raw=model_payload(&count);
	if(!raw)return NULL;
	float *packed=malloc(count*sizeof(*packed));
	if(!packed){fprintf(stderr,"brainchopc: out of memory\n");return NULL;}
#if !BC_MODEL_LOADER_TRANSPOSE
	memcpy(packed,raw,count*sizeof(*packed));
	return packed;
#else
	/* Match the compiled MindGrab table: transpose TFJS [D,H,W] taps to
	   NIfTI [x,y,z] while retaining channel-last kernel order. */
	size_t src_off=0,dst_off=0;
	for(int l=0;l<BC_MODEL_NHIDDEN;l++) {
		int cin=l?BC_MODEL_CHAN:1;
		for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
			int kd=dx+1,kh=dy+1,kw=dz+1;
			size_t src=src_off+(size_t)(kd*9+kh*3+kw)*cin*BC_MODEL_CHAN;
			memcpy(packed+dst_off,raw+src,(size_t)cin*BC_MODEL_CHAN*sizeof(float));
			dst_off+=(size_t)cin*BC_MODEL_CHAN;
		}
		src_off+=(size_t)27*cin*BC_MODEL_CHAN;
	}
	memcpy(packed+dst_off,raw+src_off,(count-src_off)*sizeof(float));
	return packed;
#endif
}

static nifti_image *copy_header(const nifti_image *source) {
	nifti_image *copy=calloc(1,sizeof(*copy));
	if(!copy)return NULL;
	*copy=*source;
	copy->fname=NULL;
	copy->iname=NULL;
	copy->data=NULL;
	copy->num_ext=0;
	copy->ext_list=NULL;
	return copy;
}

/*
 * Derive the first 8 lowercase hex characters (32 bits) of the FIPS 180-4
 * SHA-256 digest of the embedded model.bin payload. Constants are computed
 * from the standard fractional prime square/cube roots (21 bits of double
 * precision margin above 32 bits), keeping zero external dependencies and
 * leaving all CUDA/HIP manifest-locked descriptors untouched.
 */
static uint32_t bc_rotr32(uint32_t v, unsigned bits) {
	return (v >> bits) | (v << (32u - bits));
}

static void bc_sha256_block(uint32_t st[8], const uint32_t rk[64],
		const unsigned char blk[64]) {
	uint32_t msg[64];
	for (int i = 0; i < 16; i++) {
		msg[i] = ((uint32_t)blk[4 * i] << 24) | ((uint32_t)blk[4 * i + 1] << 16) |
		         ((uint32_t)blk[4 * i + 2] << 8) | (uint32_t)blk[4 * i + 3];
	}
	for (int i = 16; i < 64; i++) {
		uint32_t p0 = bc_rotr32(msg[i - 15], 7u) ^ bc_rotr32(msg[i - 15], 18u) ^ (msg[i - 15] >> 3u);
		uint32_t p1 = bc_rotr32(msg[i - 2], 17u) ^ bc_rotr32(msg[i - 2], 19u) ^ (msg[i - 2] >> 10u);
		msg[i] = msg[i - 16] + p0 + msg[i - 7] + p1;
	}
	uint32_t v0 = st[0], v1 = st[1], v2 = st[2], v3 = st[3];
	uint32_t v4 = st[4], v5 = st[5], v6 = st[6], v7 = st[7];
	for (int i = 0; i < 64; i++) {
		uint32_t u1 = bc_rotr32(v4, 6u) ^ bc_rotr32(v4, 11u) ^ bc_rotr32(v4, 25u);
		uint32_t ch = (v4 & v5) ^ ((~v4) & v6);
		uint32_t t1 = v7 + u1 + ch + rk[i] + msg[i];
		uint32_t u0 = bc_rotr32(v0, 2u) ^ bc_rotr32(v0, 13u) ^ bc_rotr32(v0, 22u);
		uint32_t mj = (v0 & v1) ^ (v0 & v2) ^ (v1 & v2);
		uint32_t t2 = u0 + mj;
		v7 = v6; v6 = v5; v5 = v4; v4 = v3 + t1;
		v3 = v2; v2 = v1; v1 = v0; v0 = t1 + t2;
	}
	st[0] += v0; st[1] += v1; st[2] += v2; st[3] += v3;
	st[4] += v4; st[5] += v5; st[6] += v6; st[7] += v7;
}

static const char *model_weights_sha8(void) {
	static char cached[9];
	if (cached[0]) return cached;
	uint32_t st[8], rk[64];
	int found = 0;
	for (int cand = 2; found < 64; cand++) {
		int is_prime = 1;
		for (int d = 2; d * d <= cand; d++) {
			if (cand % d == 0) { is_prime = 0; break; }
		}
		if (!is_prime) continue;
		if (found < 8) {
			double r2 = sqrt((double)cand);
			st[found] = (uint32_t)((r2 - floor(r2)) * 4294967296.0);
		}
		double r3 = cbrt((double)cand);
		rk[found] = (uint32_t)((r3 - floor(r3)) * 4294967296.0);
		found++;
	}
	const unsigned char *buf = bc_model_payload_begin;
	size_t total = (size_t)(bc_model_payload_end - bc_model_payload_begin);
	size_t pos = 0;
	while (pos + 64u <= total) {
		bc_sha256_block(st, rk, buf + pos);
		pos += 64u;
	}
	unsigned char pad[64];
	memset(pad, 0, sizeof(pad));
	size_t rem = total - pos;
	if (rem) memcpy(pad, buf + pos, rem);
	pad[rem] = 0x80u;
	if (rem >= 56u) {
		bc_sha256_block(st, rk, pad);
		memset(pad, 0, sizeof(pad));
	}
	uint64_t nbits = (uint64_t)total * 8u;
	for (int i = 0; i < 8; i++)
		pad[63 - i] = (unsigned char)((nbits >> (8u * (unsigned)i)) & 0xffu);
	bc_sha256_block(st, rk, pad);
	snprintf(cached, sizeof(cached), "%08x", st[0]);
	return cached;
}

static void format_provenance_descrip(char descrip[80],
		const char *backend_name, int no_turbo, int is_pve) {
	const char *model_name =
		!strcmp(BC_MODEL_NAME, "model16chan18cls") ? "mindmap-fast" : BC_MODEL_NAME;
	const char *bname = backend_name ? backend_name : "unknown";
	const char *wsha = model_weights_sha8();
	memset(descrip, 0, 80);
	if (is_pve) {
		snprintf(descrip, 80, "brainchop %s %s/%s %s pve",
			BC_VERSION, model_name, wsha, bname);
	} else if (!strcmp(bname, "metal") || !strcmp(bname, "webgpu")) {
		snprintf(descrip, 80, "brainchop %s %s/%s %s %s",
			BC_VERSION, model_name, wsha, bname, no_turbo ? "exact" : "turbo");
	} else if (!strcmp(bname, "cpu")) {
#ifdef __EMSCRIPTEN__
		snprintf(descrip, 80, "brainchop %s %s/%s %s exact",
			BC_VERSION, model_name, wsha, bname);
#else
		snprintf(descrip, 80, "brainchop %s %s/%s %s %s",
			BC_VERSION, model_name, wsha, bname, no_turbo ? "exact" : "turbo");
#endif
	} else {
		snprintf(descrip, 80, "brainchop %s %s/%s %s",
			BC_VERSION, model_name, wsha, bname);
	}
}

static int write_float_image(const nifti_image *reference, const float *values,
		const char *path, const char *backend_name, int no_turbo) {
	nifti_image *image=copy_header(reference);
	if(!image)return 1;
	image->data=(void *)values;
	image->datatype=DT_FLOAT32;
	image->nbyper=4;
	image->scl_slope=1.0;
	image->scl_inter=0.0;
	image->intent_code=NIFTI_INTENT_NONE;
	image->intent_p1=image->intent_p2=image->intent_p3=0.0;
	memset(image->intent_name,0,sizeof(image->intent_name));
	format_provenance_descrip(image->descrip,backend_name,no_turbo,0);
	int failed=nifti_set_filenames(image,path,0,1)||nifti_image_write_status(image);
	image->data=NULL;
	nifti_image_free(image);
	if(failed)fprintf(stderr,"brainchopc: unable to write %s\n",path);
	return failed;
}

typedef struct {
	const nifti_image *reference;
	const char *output;
	const char *backend_name;
	int no_turbo;
} class_output_context;

static int write_class(int class_index, const float *values, size_t count,
		void *opaque) {
	class_output_context *context=(class_output_context *)opaque;
	if(!context||!context->reference||count!=(size_t)context->reference->nvox)return 1;
	char base[4096],path[4096];
	size_t length=strlen(context->output);
	if(length>=sizeof(base))return 1;
	memcpy(base,context->output,length+1);
	/* Carry the caller's extension over. Forcing .nii here wrote 18 x 67 MB of
	   uncompressed float for model16 from a run whose main output was gzipped. */
	const char *suffix=".nii.gz";
	if(length>=7&&!strcmp(base+length-7,".nii.gz"))base[length-7]=0;
	else if(length>=8&&!strcmp(base+length-8,".nii.zst")){base[length-8]=0;suffix=".nii.zst";}
	else if(length>=4&&!strcmp(base+length-4,".nii")){base[length-4]=0;suffix=".nii";}
	if(snprintf(path,sizeof(path),"%s_c%d%s",base,class_index,suffix)>=(int)sizeof(path))
		return 1;
	if(write_float_image(context->reference,values,path,
			context->backend_name,context->no_turbo))return 1;
	fprintf(stderr,"brainchop :: Saved %s\n",path);
	return 0;
}

static int write_mask(const nifti_image *reference, const float *values,
		const char *path, const char *backend_name, int no_turbo) {
	nifti_image *mask=copy_header(reference);
	if(!mask)return 1;
	unsigned char *data=nii_malloc((size_t)reference->nvox,1);
	if(!data){nifti_image_free(mask);return 1;}
	for(size_t i=0;i<(size_t)reference->nvox;i++)data[i]=values[i]>0.0f?1:0;
	mask->data=data;
	mask->datatype=DT_UINT8;
	mask->nbyper=1;
	mask->scl_slope=1.0;
	mask->scl_inter=0.0;
	mask->cal_min=0.0;
	mask->cal_max=0.0;
	mask->intent_code=NIFTI_INTENT_LABEL;
	mask->intent_p1=mask->intent_p2=mask->intent_p3=0.0;
	memset(mask->intent_name,0,sizeof(mask->intent_name));
	format_provenance_descrip(mask->descrip,backend_name,no_turbo,0);
	int failed=nifti_set_filenames(mask,path,0,1)||nifti_image_write_status(mask);
	nifti_image_free(mask);
	if(failed)fprintf(stderr,"brainchopc: unable to write %s\n",path);
	else fprintf(stderr,"brainchop :: Saved %s\n",path);
	return failed;
}

#if BC_HAS_PVE
/*
 * One tissue-fraction map.
 *
 * Resliced to the input grid unless --save-conform, exactly as the label path
 * is, so `--pve` obeys the same output-space default as every other mode --
 * with LINEAR interpolation, because a continuous fraction is not a label.
 * Written as uint8 with scl_slope 1/255 so a reader recovers [0, 1]: a
 * deliberate 4x size cut, and 1/255 is finer than the fit's own agreement with
 * the reference.
 */
static int write_fraction(const nifti_image *conformed, const nifti_image *original,
		const float *values, const char *output, const char *tissue,
		int save_conform, const char *backend_name, int no_turbo) {
	char base[4096],path[4096];
	size_t length=strlen(output);
	if(length>=sizeof(base))return 1;
	memcpy(base,output,length+1);
	/* Carry the caller's extension over, as write_class does and for the same
	   reason: forcing .nii here would write uncompressed float from a run whose
	   main output was gzipped. */
	const char *suffix=".nii.gz";
	if(length>=7&&!strcmp(base+length-7,".nii.gz"))base[length-7]=0;
	else if(length>=8&&!strcmp(base+length-8,".nii.zst")){base[length-8]=0;suffix=".nii.zst";}
	else if(length>=4&&!strcmp(base+length-4,".nii")){base[length-4]=0;suffix=".nii";}
	if(snprintf(path,sizeof(path),"%s_%s%s",base,tissue,suffix)>=(int)sizeof(path))
		return 1;

	nifti_image *image=copy_header(conformed);
	if(!image)return 1;
	float *data=nii_malloc((size_t)conformed->nvox,sizeof(*data));
	if(!data){nifti_image_free(image);return 1;}
	memcpy(data,values,(size_t)conformed->nvox*sizeof(*data));
	image->data=data;
	image->datatype=DT_FLOAT32;
	image->nbyper=(int)sizeof(float);
	image->scl_slope=1.0f;
	image->scl_inter=0.0;
	if(!save_conform&&brainchopc_reslice(image,original,1)) {
		fprintf(stderr,"brainchopc: cannot reslice to the input grid; its "
			"sform is not a usable spatial transform\n");
		nifti_image_free(image);return 1;
	}

	/* Quantise in place, AFTER the reslice: interpolating uint8 codes would
	   round twice. */
	const float *resliced=(const float *)image->data;
	unsigned char *quantised=nii_malloc((size_t)image->nvox,1);
	if(!quantised){nifti_image_free(image);return 1;}
	for(size_t i=0;i<(size_t)image->nvox;i++) {
		float v=resliced[i];
		if(!(v>0.0f))v=0.0f;
		if(v>1.0f)v=1.0f;
		quantised[i]=(unsigned char)(v*255.0f+0.5f);
	}
	free(image->data);
	image->data=quantised;
	image->datatype=DT_UINT8;
	image->nbyper=1;
	image->scl_slope=1.0f/255.0f;
	image->scl_inter=0.0;
	image->cal_min=0.0;
	image->cal_max=1.0;
	image->intent_code=NIFTI_INTENT_NONE;
	image->intent_p1=image->intent_p2=image->intent_p3=0.0;
	memset(image->intent_name,0,sizeof(image->intent_name));
	format_provenance_descrip(image->descrip,backend_name,no_turbo,1);
	int failed=nifti_set_filenames(image,path,0,1)||nifti_image_write_status(image);
	nifti_image_free(image);
	if(failed)fprintf(stderr,"brainchopc: unable to write %s\n",path);
	else fprintf(stderr,"brainchop :: Saved %s\n",path);
	return failed;
}
#endif

/* Treat unusable categorical output as inference failure, so automatic backend
   fallback follows the same policy as allocation/dispatch failures. */
static int segment_checked(const bc_backend *backend, const nifti_image *image,
		float *result, const float *weights, const bc_inference_options *options) {
	if (options->tissue_priors_prepared) *options->tissue_priors_prepared = 0;
	int failed = backend->segment(image->data, result, weights, options);
	if (!failed && !BC_MODEL_OUTPUT_IS_MASK && !options->tissue_planes && !options->class_writer) {
		const size_t dim[3] = {(size_t)image->nx, (size_t)image->ny, (size_t)image->nz};
		failed = bwlabel_validate(result, dim, BC_MODEL_NCLASS);
	}
	/* An empty mask is a failure, not a warning: it is what a GPU stack that
	   silently drops draws returns (SwiftShader, brainchop#78). */
	if (!failed && BC_MODEL_OUTPUT_IS_MASK && !options->class_writer) {
		size_t foreground = 0;
		for (size_t i = 0; i < (size_t)image->nvox && !foreground; i++) foreground = result[i] > 0.0f;
		if (!foreground) {
			fprintf(stderr, "mindgrab: the network found no brain; check the input contrast\n");
			failed = 1;
		}
	}
	return failed;
}

static int run_model(nifti_image *original, int input_datatype,
		int mindgrab_output, const float *weights, const bc_backend *backend,
		int allow_fallback, const bc_cli_options *cli, nifti_image **output) {
	*output=NULL;
	bc_stage(NULL);
	nifti_image *conformed=private_float_copy(original);
	if (!conformed) return 1;
	bc_stage("copy");
	if(cli->ct&&brainchopc_hounsfield_to_cormack(conformed)) {
		fprintf(stderr,"brainchopc: CT conversion failed\n");
		nifti_image_free(conformed);return 1;
	}
	if(cli->comply&&brainchopc_comply(conformed)) {
		fprintf(stderr,"brainchopc: compliance transform failed\n");
		nifti_image_free(conformed);return 1;
	}
	if (brainchopc_conform(conformed)) {
		fprintf(stderr,"brainchopc: conform failed\n");
		nifti_image_free(conformed);
		return 1;
	}
	bc_stage("conform");
	if (brainchopc_convert(conformed,DT_UINT8)) {
		nifti_image_free(conformed);
		return 1;
	}
	bc_inference_options inference={0};
	inference.force_tuning=cli->tune;
	/* Not the {0} above: 0 is a valid adapter index, -1 is "backend chooses". */
	inference.adapter=cli->adapter;
	inference.no_turbo=cli->no_turbo;
	if(cli->crop) {
		bc_crop_box box;
		if(brainchopc_crop_box(conformed->data,(int)conformed->nx,
				(int)conformed->ny,(int)conformed->nz,cli->crop_percent,&box)||
				box.x_high<box.x_low) {
			fprintf(stderr,"brainchopc: --crop found no foreground voxels\n");
			nifti_image_free(conformed);return 1;
		}
		inference.roi_enabled=1;
		inference.x_low=box.x_low; inference.x_high=box.x_high;
		inference.y_low=box.y_low; inference.y_high=box.y_high;
		inference.z_low=box.z_low; inference.z_high=box.z_high;
	}
	class_output_context class_context={conformed,cli->output,backend->name,cli->no_turbo};
	if(cli->export_classes) {
		inference.class_writer=write_class;
		inference.class_writer_context=&class_context;
	}
	float *result=nii_malloc((size_t)conformed->nvox,sizeof(*result));
	int failed=!result;
#if BC_HAS_PVE
	float *planes=NULL;
	int priors_prepared=0;
	if(!failed&&cli->pve) {
		planes=nii_malloc((size_t)conformed->nvox*BC_PVE_PLANES,sizeof(*planes));
		if(!planes) {
			fprintf(stderr,"brainchopc: out of memory (--pve needs about %.1f GB "
				"beyond the network)\n",
				(double)conformed->nvox*BC_PVE_PLANES*sizeof(float)/1e9);
			failed=1;
		}
		inference.tissue_planes=planes;
		inference.tissue_priors_prepared=&priors_prepared;
	}
#endif
	bc_stage("to uint8 + setup");
	const bc_backend *effective_backend=backend;
	if(!failed) failed=segment_checked(effective_backend,conformed,result,weights,&inference);
	bc_stage("segment (incl GPU)");
	if(failed && allow_fallback && effective_backend!=&bc_cpu_backend) {
		fprintf(stderr,"brainchopc: %s backend failed; falling back to CPU\n",effective_backend->name);
		effective_backend=&bc_cpu_backend;
		class_context.backend_name=effective_backend->name;
		failed=segment_checked(effective_backend,conformed,result,weights,&inference);
	}
	if (failed) {
#if BC_HAS_PVE
		free(planes);
#endif
		free(result);
		nifti_image_free(conformed);
		return 1;
	}
	size_t dimensions[3]={(size_t)conformed->nx,(size_t)conformed->ny,
		(size_t)conformed->nz};
#if BC_HAS_PVE
	if(cli->pve) {
		float *intensity=bc_accel_normalize_input(conformed->data);
		int pve_failed=!intensity;
		if(!pve_failed&&!priors_prepared)pve_failed=bc_cat_lite_priors(planes,dimensions);
		bc_stage("pve priors");
		/* BRAINCHOPC_PVE_DUMP: the blurred, support-gated priors and the
		   normalized intensity, raw f32, at exactly the boundary cat-lite.js
		   takes as input. Feeding both implementations this file is what
		   tests the port rather than the network in front of it. */
		const char *dump=getenv("BRAINCHOPC_PVE_DUMP");
		if(!pve_failed&&dump&&*dump) {
			FILE *handle=fopen(dump,"wb");
			if(!handle||
					fwrite(planes,sizeof(float),(size_t)conformed->nvox*BC_PVE_PLANES,handle)!=
						(size_t)conformed->nvox*BC_PVE_PLANES||
					fwrite(intensity,sizeof(float),(size_t)conformed->nvox,handle)!=
						(size_t)conformed->nvox) {
				fprintf(stderr,"brainchopc: unable to write %s\n",dump);
				pve_failed=1;
			}
			/* Checked: a 335 MB write that fails only at flush would
			   otherwise report success and leave a truncated dump, and the
			   gate would blame the port for a full disk. */
			if(handle&&fclose(handle)&&!pve_failed) {
				fprintf(stderr,"brainchopc: unable to flush %s\n",dump);
				pve_failed=1;
			}
			if(!pve_failed)fprintf(stderr,"brainchop :: Dumped priors to %s\n",dump);
		}
		if(!pve_failed)pve_failed=bc_cat_lite_fit(planes,intensity,dimensions);
		bc_stage("pve fit");
		free(intensity);
		static const char *const tissue[3]={"gm","wm","csf"};
		for(int i=0;i<3&&!pve_failed;i++)
			pve_failed=write_fraction(conformed,original,
				planes+(size_t)i*conformed->nvox,cli->output,tissue[i],
				cli->save_conform,effective_backend->name,cli->no_turbo);
		/* brain = GM + WM combined continuous tissue fraction map, summed in place. */
		for(size_t v=0;v<(size_t)conformed->nvox&&!pve_failed;v++)
			planes[v]+=planes[(size_t)conformed->nvox+v];
		if(!pve_failed)
			pve_failed=write_fraction(conformed,original,planes,cli->output,
				"brain",cli->save_conform,effective_backend->name,cli->no_turbo);
		bc_stage("pve write");
		free(planes);free(result);nifti_image_free(conformed);
		return pve_failed;
	}
#endif
	/* Optional raw label capture lets the independent browser implementation
	   validate cleanup on exactly the same network output. */
	const char *label_dump=getenv("BRAINCHOPC_LABEL_DUMP");
	if(label_dump&&*label_dump) {
		FILE *handle=fopen(label_dump,"wb");
		int error=!handle;
		if(handle) {
			error=fwrite(result,sizeof(float),(size_t)conformed->nvox,handle)!=
				(size_t)conformed->nvox;
			if(fclose(handle))error=1;
		}
		if(error) {
			fprintf(stderr,"brainchopc: unable to write %s\n",label_dump);
			free(result);nifti_image_free(conformed);return 1;
		}
	}
	int cleanup_failed;
#if defined(BC_MODEL_IS_MINDMAP) || defined(BC_MODEL_IS_MODEL16CHAN18CLS)
	cleanup_failed=cli->legacy_cleanup ? bwlabel_largest(result,dimensions,0,26) :
		bwlabel_cleanup18(result,dimensions);
#elif defined(BC_MODEL_IS_MINDSNAP)
	cleanup_failed=cli->legacy_cleanup ? bwlabel_largest(result,dimensions,0,26) :
		bwlabel_cleanup104(result,dimensions);
#else
	cleanup_failed=bwlabel_largest(result,dimensions,BC_MODEL_OUTPUT_IS_MASK,26);
#endif
	if(cleanup_failed) {
		free(result);nifti_image_free(conformed);return 1;
	}
	if(mindgrab_output&&cli->border_mm>0&&
			brainchopc_grow_mask(result,(int)conformed->nx,(int)conformed->ny,
				(int)conformed->nz,(float)conformed->dx,(float)conformed->dy,
				(float)conformed->dz,(float)cli->border_mm)) {
		fprintf(stderr,"brainchopc: unable to grow mask border\n");
		free(result);nifti_image_free(conformed);return 1;
	}
	bc_stage("bwlabel + border");
	free(conformed->data);
	conformed->data=result; conformed->datatype=DT_FLOAT32; conformed->nbyper=4;
	if(mindgrab_output) {
		if(brainchopc_reslice(conformed,original,0)) {
			fprintf(stderr,"brainchopc: cannot reslice to the input grid; its "
				"sform is not a usable spatial transform\n");
			nifti_image_free(conformed);return 1;
		}
		float *img=original->data,*resliced=conformed->data;
		if(cli->mask_output&&write_mask(original,resliced,cli->mask_output,
				effective_backend->name,cli->no_turbo)) {
			nifti_image_free(conformed);return 1;
		}
		float mn=INFINITY;
		for(size_t i=0;i<(size_t)original->nvox;i++)mn=fminf(mn,img[i]);
		for(size_t i=0;i<(size_t)original->nvox;i++)
			if(resliced[i]<=0.0f)img[i]=mn;
		nifti_image_free(conformed);
		if(brainchopc_convert(original,input_datatype))return 1;
		bc_stage("reslice + mask");
		format_provenance_descrip(original->descrip,effective_backend->name,cli->no_turbo,0);
		*output=original;
		return 0;
	}

	/*
	 * Label models write their output in the INPUT image's own space by
	 * default, matching what a caller almost always wants: a label map aligned
	 * to the image they supplied. --save-conform keeps the conformed 256^3
	 * volume instead. This inverts the older behaviour, where conformed was the
	 * default and --inverse-conform opted into the input grid.
	 */
	conformed->intent_code=NIFTI_INTENT_LABEL;
	conformed->intent_p1=conformed->intent_p2=conformed->intent_p3=0.0;
	memset(conformed->intent_name,0,sizeof(conformed->intent_name));
	if(!cli->save_conform&&brainchopc_reslice(conformed,original,BC_RESAMPLE_LABELS)) {
		fprintf(stderr,"brainchopc: cannot reslice to the input grid; its "
			"sform is not a usable spatial transform\n");
		nifti_image_free(conformed);return 1;
	}
	if(brainchopc_convert(conformed,DT_UINT8)) {
		nifti_image_free(conformed);return 1;
	}
	bc_stage("reslice + convert");
	format_provenance_descrip(conformed->descrip,effective_backend->name,cli->no_turbo,0);
	*output=conformed;
	return 0;
}

int main(int argc, char **argv) {
	bc_cli_options cli;
	int parse_status=bc_cli_parse(argc,argv,&cli);
	if(parse_status)return parse_status==1?0:2;
	const char *backend_name=cli.backend;
#ifdef BC_THREADS
	/* Passed raw. bc_threads.c's resolve_count() owns the clamp -- this used to
	   reject anything over INT_MAX with exit 2 while the pool silently clamped
	   everything else, so one question had two answers depending on how large
	   the number was. */
	if(cli.cpu_threads>0) bc_threads_set_count(cli.cpu_threads);
#else
	if(cli.cpu_threads>0)
		fprintf(stderr,"brainchopc: warning: -cpu has no effect in this serial build\n");
#endif
	float *weights=load_model_weights();
	if(!weights)return 1;
	bc_stage(NULL);
	nifti_image *nim=nifti_image_read(cli.input,1);
	bc_stage("read + gunzip");
	if (!nim) { fprintf(stderr,"brainchopc: unable to read %s\n",cli.input); free(weights);return 1; }
	if (nim->nx<1 || nim->ny<1 || nim->nz<1 ||
	    nim->nvox != nim->nx*nim->ny*nim->nz) {
		fprintf(stderr,"brainchopc: input must be a scalar 3D image\n");
		free(weights);nifti_image_free(nim); return 1;
	}
	int input_datatype=nim->datatype;
	int mindgrab_output=BC_MODEL_OUTPUT_IS_MASK;
#if BC_HAS_PVE
	if(cli.pve&&cli.export_classes) {
		fprintf(stderr,"brainchopc: --pve and --export-classes ask for different "
			"outputs; pick one\n");
		free(weights);nifti_image_free(nim);return 2;
	}
#endif
	int special_cpu=cli.crop||cli.export_classes;
	int explicit_gpu=backend_name&&*backend_name&&strcmp(backend_name,"auto")&&
		strcmp(backend_name,"cpu");
	if(special_cpu&&explicit_gpu&&!cli.cpu_requested) {
		fprintf(stderr,"brainchopc: --crop and --export-classes "
			"require the CPU backend\n");
		free(weights);nifti_image_free(nim);return 2;
	}
	int force_cpu=cli.cpu_requested||special_cpu||
		(backend_name&&!strcmp(backend_name,"cpu"));
	int automatic_backend=!backend_name||!*backend_name||!strcmp(backend_name,"auto");
	const bc_backend *backend=bc_backend_select(force_cpu,backend_name);
#if BC_HAS_PVE
	/* PVE requires a backend that fills all four tissue planes. Automatic
	 * selection falls back to CPU when its preferred GPU has no such head;
	 * an explicit unsupported GPU must fail, never consume uninitialized data. */
	if (cli.pve && backend && backend != &bc_cpu_backend
#ifdef HAVE_CUDA
		&& backend != &bc_cuda_backend
#endif
#ifdef HAVE_HIP
		&& backend != &bc_hip_backend
#endif
#ifdef HAVE_WEBGPU
		&& backend != &bc_webgpu_backend
#endif
#ifdef HAVE_WEBGL2
		&& backend != &bc_webgl2_backend
#endif
#ifdef HAVE_METAL
		&& backend != &bc_metal_backend
#endif
		) {
		if (explicit_gpu && !cli.cpu_requested) {
			fprintf(stderr,"brainchopc: --pve requires a backend with tissue priors (CPU, CUDA, HIP, Metal, WebGPU or WebGL2)\n");
			free(weights);nifti_image_free(nim);return 2;
		}
		backend=&bc_cpu_backend;
	}
#endif

	if(!backend) {
		/* A backend the Makefile left out of THIS executable is a different
		 * failure from a machine with no usable device; say which. */
		const char *excluded=NULL;
#ifdef BC_OPENCL_EXCLUDED
		if(backend_name&&!strcmp(backend_name,"opencl"))
			excluded="its channel count exceeds the OpenCL backend's 16-channel limit";
#endif
#ifdef BC_CUDA_EXCLUDED
		if(backend_name&&!strcmp(backend_name,"cuda"))
			excluded="no CUDA module is built for this model";
#endif
		if(excluded)
			fprintf(stderr,"brainchopc: backend '%s' is not built into this executable: %s\n",
				backend_name,excluded);
		else
			fprintf(stderr,"brainchopc: requested backend '%s' is unavailable\n",
				backend_name?backend_name:"");
		free(weights);nifti_image_free(nim);return 1;
	}
	fprintf(stderr,"brainchop :: [1/1] %s\n",cli.input);
	fprintf(stderr,"brainchopc: v%s, using %s backend\n",BC_VERSION,backend->name);
	/*
	 * --tune only means something to a backend that searches its kernel shape,
	 * and the selected backend may not be one -- either because the user asked
	 * for another, or because a flag like --crop forced the CPU. Say so rather
	 * than accepting the option and doing nothing, which is the one behaviour
	 * the rest of this CLI is careful to avoid.
	 */
	if(cli.tune && !BC_TUNABLE_BACKEND(backend->name))
		fprintf(stderr,"brainchopc: --tune has no effect on the %s backend; "
			"it does not tune its kernel shape\n",backend->name);
	/*
	 * Validate --adapter HERE, against the backend that was actually selected,
	 * and before conform and preprocessing run. The index is only meaningful
	 * relative to a backend -- "1" is different hardware to CUDA than to OpenCL
	 * -- so it cannot be checked at parse time, when the backend is still a
	 * string. Checking it late enough to be correct but early enough to cost
	 * nothing is the point: the alternative is failing after a minute of work.
	 */
	if(!bc_backend_adapter_valid(backend,cli.adapter)) {
		free(weights);nifti_image_free(nim);return 1;
	}
	nifti_image *output=NULL;
	if (brainchopc_convert(nim,DT_FLOAT32) || sanitize_input(nim,input_datatype) ||
	    run_model(nim,input_datatype,mindgrab_output,weights,backend,automatic_backend,
			&cli,&output)) {
		free(weights);nifti_image_free(nim); return 1;
	}
	free(weights);
	bc_stage(NULL);
	if(output!=nim)nifti_image_free(nim);
	/*
	 * --pve has already written its three tissue maps and has no label volume
	 * to save, so run_model succeeds with no image. Everything below assumes
	 * one: without this the next line dereferences NULL, which reached the
	 * gate as a segfault AFTER all three files had been written and reported.
	 */
	if(!output) {
		bc_stage("write + gzip");
		fprintf(stderr,"brainchop :: Done!\n");
		return 0;
	}
	if (nifti_set_filenames(output,cli.output,0,1) ||
	    nifti_image_write_status(output)) {
		fprintf(stderr,"brainchopc: unable to write %s\n",cli.output);
		nifti_image_free(output); return 1;
	}
	bc_stage("write + gzip");
	fprintf(stderr,"brainchop :: Saved %s\nbrainchop :: Done!\n",cli.output);
	nifti_image_free(output);
	return 0;
}
