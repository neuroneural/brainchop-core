#ifndef BRAINCHOPC_CLI_H
#define BRAINCHOPC_CLI_H

#include "cat_lite.h"
#include "model_specs.h"
#include <stdio.h>
#include <string.h>

typedef struct {
	const char *input;
	const char *output;
	const char *mask_output;
	const char *backend;
	long cpu_threads;
	double crop_percent;
	int cpu_requested;
	int ct;
	int crop;
	int comply;
	/*
	 * Label models write their output in the input image's own space by
	 * default; this asks for the conformed 256^3 grid instead. It is the
	 * inverse of the old --inverse-conform, which opted into native space from
	 * a conformed default.
	 */
	int save_conform;
	int export_classes;
	int legacy_cleanup; /* Label models: original 26-connected binary LCC. */
#if BC_HAS_PVE
	/*
	 * --pve: write continuous GM/WM/CSF tissue fractions instead of a
	 * label volume. CPU/GPU tissue-prior backends, and rejected at parse time
	 * against anything it contradicts rather than resolved downstream.
	 */
	int pve;
#endif
	int border_mm;
	/* --tune: re-measure the GPU kernel shape even if a cached one exists. */
	int tune;
	/*
	 * --adapter: which GPU to run on, as an index into the chosen backend's own
	 * enumeration, or -1 for "the backend decides". Defaults to -1, so a build
	 * without the option behaves exactly as it did before.
	 */
	int adapter;
	/* --no-turbo: exact CPU / fp16 GPU instead of the turbo default. */
	int no_turbo;
} bc_cli_options;

/*
 * Which backends tune their kernel shape, in ONE place.
 *
 * This fact previously had two owners: cli.c gated the --tune flag on
 * `defined(HAVE_CUDA)` while brainchopc.c independently tested
 * `strcmp(backend->name, "cuda")` to decide whether to warn. Adding OpenCL to
 * only the first -- which cli.c's own comment invites, and which this design
 * names as the next task -- would accept the flag, tune correctly, and THEN
 * print that it had no effect. Both live here so they move together.
 */
#define BC_TUNABLE_BACKEND(name) \
	(!strcmp((name), "cuda") || !strcmp((name), "opencl"))
#if defined(HAVE_CUDA) || defined(HAVE_OPENCL)
#define BC_HAS_TUNABLE_BACKEND 1
#else
#define BC_HAS_TUNABLE_BACKEND 0
#endif

/* Label models with hierarchical browser-parity cleanup offer --legacy-cleanup. */
#define BC_HAS_LEGACY_CLEANUP (!BC_MODEL_OUTPUT_IS_MASK)

/* Usage text for this build, listing only the options it can honour. */
void bc_cli_help(FILE *stream, const char *program);

/* Returns 0 to run, 1 after a successful informational action, or 2 on error. */
int bc_cli_parse(int argc, char **argv, bc_cli_options *options);

#endif
