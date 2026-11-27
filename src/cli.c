#include "cli.h"
/* For bc_backend_list_adapters: `-a -1` is an informational action that answers
   itself here, exactly as --help does, rather than setting a flag for main. */
#include "backend.h"
#include "model_specs.h"
#include "version.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/*
 * Only the backends this build actually contains. A WebAssembly build has
 * neither Metal, CUDA nor OpenCL, and offering them in --help would be the same
 * mistake as advertising an option the model does not support.
 */
#if defined(HAVE_METAL)
#define BC_BACKEND_METAL "|metal"
#else
#define BC_BACKEND_METAL ""
#endif
#if defined(HAVE_CUDA)
#define BC_BACKEND_CUDA "|cuda"
#else
#define BC_BACKEND_CUDA ""
#endif
#if defined(HAVE_OPENCL)
#define BC_BACKEND_OPENCL "|opencl"
#else
#define BC_BACKEND_OPENCL ""
#endif
#if defined(HAVE_HIP)
#define BC_BACKEND_HIP "|hip"
#else
#define BC_BACKEND_HIP ""
#endif
#if defined(HAVE_WEBGPU)
#define BC_BACKEND_WEBGPU "|webgpu"
#else
#define BC_BACKEND_WEBGPU ""
#endif
#if defined(HAVE_WEBGL2)
#define BC_BACKEND_WEBGL2 "|webgl2"
#else
#define BC_BACKEND_WEBGL2 ""
#endif
#define BC_BACKEND_CHOICES \
	"auto|cpu" BC_BACKEND_CUDA BC_BACKEND_HIP BC_BACKEND_OPENCL BC_BACKEND_METAL \
	BC_BACKEND_WEBGPU BC_BACKEND_WEBGL2
/* bc_backend_select's auto preference order; the leading '|' is skipped.
   OpenCL is opt-in only (-backend opencl), never auto-selected. */
#define BC_BACKEND_AUTO_ORDER \
	BC_BACKEND_CUDA BC_BACKEND_HIP BC_BACKEND_METAL BC_BACKEND_WEBGPU \
	BC_BACKEND_WEBGL2 "|cpu"

/* BC_HAS_TUNABLE_BACKEND lives in cli.h, beside the name test
   brainchopc.c uses, so the two cannot drift apart. */

/*
 * Whether this build contains a backend that can enumerate and choose between
 * several adapters, which is what --adapter drives. Metal and WebGPU could join
 * by filling in the two adapter hooks in their bc_backend and being added here.
 */
#if defined(HAVE_CUDA) || defined(HAVE_HIP) || defined(HAVE_OPENCL)
#define BC_HAS_SELECTABLE_ADAPTER 1
#else
#define BC_HAS_SELECTABLE_ADAPTER 0
#endif

/*
 * The help text is assembled from this build's capabilities rather than being
 * one fixed string, so an executable advertises exactly the options it can
 * honour and nothing else. Somebody reading --help should never find an option
 * that this binary will then refuse.
 */
void bc_cli_help(FILE *stream, const char *program) {
	int pad = (int)strlen(program);
	fprintf(stream, "usage: %s [-h] [-v] [-o OUTPUT] [--ct] [--comply]\n", program);
	fprintf(stream, "       %*s [--export-classes]", pad, "");
#if BC_HAS_PVE
	fprintf(stream, " [--pve]");
#endif
#if BC_HAS_LEGACY_CLEANUP
	fprintf(stream, " [--legacy-cleanup]");
#endif
#if BC_SUPPORTS_CROP
	fprintf(stream, " [--crop [PCT]]");
#endif
#if BC_SUPPORTS_SAVE_CONFORM
	fprintf(stream, " [--save-conform]");
#endif
#if BC_SUPPORTS_MASK
	fprintf(stream, " [--mask [PATH]]");
#endif
#if BC_SUPPORTS_BORDER
	fprintf(stream, " [--border MM]");
#endif
#if BC_HAS_SELECTABLE_ADAPTER
	fprintf(stream, " [-a N]");
#endif
#if BC_HAS_TUNABLE_BACKEND
	/* Every other capability-gated flag appears in BOTH the synopsis and the
	   options body; --tune was wired into only the body. */
	fprintf(stream, " [--tune]");
#endif
	fprintf(stream, " [--no-turbo]");
	fprintf(stream, "\n       %*s [-cpu N] [-backend %s] input\n\n",
		pad, "", BC_BACKEND_CHOICES);

	const char *public_name =
		!strcmp(BC_MODEL_NAME, "model16chan18cls") ? "mindmap-fast" :
		!strcmp(BC_MODEL_NAME, "mindmap_omni") ? "mindmap-omni" : BC_MODEL_NAME;
	const char *public_desc =
		!strcmp(BC_MODEL_NAME, "mindgrab") ? "skull stripping in any modality" :
		!strcmp(BC_MODEL_NAME, "mindmap") ? "18-class cortical and subcortical segmentation" :
		!strcmp(BC_MODEL_NAME, "mindmap_omni") ? "contrast-agnostic 18-class segmentation (T2w, FLAIR, T1w, ULF, PET)" :
		!strcmp(BC_MODEL_NAME, "mindsnap") ? "104-class Desikan-Killiany parcellation" :
		!strcmp(BC_MODEL_NAME, "model16chan18cls") ? "fast lightweight 18-class segmentation" :
		BC_MODEL_DESCRIPTION;
	fprintf(stream, "BrainChop %s: %s (v%s)\n\n",
		public_name, public_desc, BC_VERSION);

	fprintf(stream,
		"positional arguments:\n"
		"  input                 Input NIfTI file\n\n"
		"options:\n"
		"  -h, --help            show this help message and exit\n"
		"  -v, --version         show the version and exit\n"
		"  -o OUTPUT, --output OUTPUT\n"
		"                        Output path (default: output.nii.gz)\n"
		"  --ct                  CT scan: convert Hounsfield to Cormack units\n"
		"  --comply              Insert compliance transform before conform\n"
		"  --export-classes      Export per-class raw maps as separate files\n");
#if BC_HAS_PVE
	fprintf(stream,
		"  --pve                 Experimental GM/WM/CSF partial-volume tissue fractions;\n"
		"                        writes OUTPUT_gm, OUTPUT_wm, OUTPUT_csf and\n"
		"                        OUTPUT_brain (GM + WM)\n");
#ifdef HAVE_OPENCL
	fprintf(stream,
		"                        OpenCL lacks tissue priors: auto uses CPU\n");
#endif
#endif
#if BC_SUPPORTS_CROP
	fprintf(stream,
		"  --crop [PCT]          Crop to brain bounding box (default percentile: 2.0)\n");
#endif
#if BC_HAS_LEGACY_CLEANUP
	fprintf(stream,
		"  --legacy-cleanup      Use the original whole-brain cleanup instead of\n"
		"                        per-class filtering and neighboring-label refill\n");
#endif
#if BC_SUPPORTS_SAVE_CONFORM
	fprintf(stream,
		"  --save-conform        Write the conformed 256^3 volume instead of\n"
		"                        reslicing back to the input grid\n");
#endif
#if BC_SUPPORTS_MASK
	fprintf(stream,
		"  --mask [PATH]         Save binary brain mask (default: mask.nii.gz)\n");
#endif
#if BC_SUPPORTS_BORDER
	fprintf(stream,
		"  --border MM           Grow mask border in mm (default: 0)\n");
#endif
#if BC_HAS_TUNABLE_BACKEND
	fprintf(stream,
		"  --tune                Re-measure the GPU kernel shape for this device and\n"
		"                        cache it. Happens automatically the first time.\n");
#endif
	fprintf(stream,
		"  --no-turbo            Exact CPU / fp16 GPU instead of the faster, approximate\n"
		"                        default (~2x slower; GPU needs ~3x the memory)\n");
#if BC_HAS_SELECTABLE_ADAPTER
	fprintf(stream,
		"  -a N, --adapter N     Select GPU adapter N for the chosen backend.\n"
		"                        -a -1 lists the adapters and their indices, then exits.\n"
		"                        Indices are per backend; see the listing headings.\n");
#endif
	fprintf(stream,
		"  -cpu N                Force CPU; N > 0 sets threads, N <= 0 selects automatically\n"
		"  -backend NAME, --backend NAME\n"
		"                        Select %s (default: auto, first available of\n"
		"                        %s)\n\n"
		"Examples:\n", BC_BACKEND_CHOICES, &BC_BACKEND_AUTO_ORDER[1]);
	fprintf(stream, "  %s input.nii.gz -o output.nii.gz\n", program);
#if BC_SUPPORTS_MASK
	fprintf(stream, "  %s input.nii.gz --ct --mask brain_mask.nii.gz\n", program);
#endif
#if BC_SUPPORTS_SAVE_CONFORM
	fprintf(stream, "  %s input.nii.gz --save-conform -o conformed.nii.gz\n", program);
#endif
}

static int parse_long(const char *text, long *value) {
	char *end = NULL;
	errno = 0;
	long parsed = strtol(text, &end, 10);
	if (errno || !end || *end) return 1;
	*value = parsed;
	return 0;
}

/*
 * Guarded by the same capability as its only caller, --crop's optional-value
 * parse. Without the guard an executable whose model has no --crop (model16)
 * compiles it dead and warns, which is noise that hides a real warning.
 */
#if BC_SUPPORTS_CROP
static int parse_double(const char *text, double *value) {
	char *end = NULL;
	errno = 0;
	double parsed = strtod(text, &end);
	if (errno || !end || *end) return 1;
	*value = parsed;
	return 0;
}
#endif

static int needs_value(int index, int argc, const char *option) {
	if (index + 1 < argc) return 0;
	fprintf(stderr, "brainchop: %s requires a value\n", option);
	return 1;
}

/*
 * Options this executable does not offer, but that a sibling executable or an
 * earlier release did. These get a message naming the reason rather than the
 * generic "unrecognized argument", because the likeliest way to arrive here is
 * carrying a command line over from the other executable, or from a script
 * written against the old CLI, and the generic error helps with neither.
 */
static int rejected_option(const char *argument) {
	static const struct {
		const char *name;
		int offered;
		const char *why;
	} known[] = {
		{ "--crop", BC_SUPPORTS_CROP,
		  "this model's receptive field spans the whole volume, so cropping\n"
		  "  would feed it mostly zero padding" },
		{ "--mask", BC_SUPPORTS_MASK,
		  "it applies only to a model whose output is a brain mask" },
		{ "--border", BC_SUPPORTS_BORDER,
		  "it applies only to a model whose output is a brain mask" },
		{ "--save-conform", BC_SUPPORTS_SAVE_CONFORM,
		  "this model's output is the input image with non-brain voxels\n"
		  "  removed, so it is always in the input's own space" },
		{ "--inverse-conform", 0,
		  "reslicing to the input grid is now the default; use --save-conform\n"
		  "  to keep the conformed 256^3 volume instead" },
		{ "-m", 0, "this executable contains a single model, " BC_MODEL_NAME },
		{ "--model", 0, "this executable contains a single model, " BC_MODEL_NAME },
		{ "-l", 0, "this executable contains a single model, " BC_MODEL_NAME },
		{ "--list", 0, "this executable contains a single model, " BC_MODEL_NAME },
		{ "--tune", BC_HAS_TUNABLE_BACKEND,
		  "this build has no GPU backend whose kernel shape can be tuned" },
		{ "--turbo", 0, "turbo is the default; --no-turbo disables it" },
		{ "-a", BC_HAS_SELECTABLE_ADAPTER,
		  "this build has no GPU backend with adapters to choose between" },
		{ "--adapter", BC_HAS_SELECTABLE_ADAPTER,
		  "this build has no GPU backend with adapters to choose between" },
		{ "--pve", BC_HAS_PVE,
		  "partial-volume tissue fractions (--pve) are supported only for mindmap" },
		{ "--partial-volume-estimate", BC_HAS_PVE,
		  "partial-volume tissue fractions (--pve) are supported only for mindmap" },
		{ "--legacy-cleanup", BC_HAS_LEGACY_CLEANUP,
		  "it applies only to label models with per-class cleanup" },
		{ "--custom", 0, "the model is compiled into this executable" },
		{ "--skull-strip", 0, "the model is compiled into this executable" },
	};
	for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
		if (known[i].offered || strcmp(argument, known[i].name)) continue;
		fprintf(stderr, "brainchop: %s is not available here: %s\n",
			argument, known[i].why);
		return 1;
	}
	return 0;
}

int bc_cli_parse(int argc, char **argv, bc_cli_options *options) {
	if (!options) return 2;
	memset(options, 0, sizeof(*options));
	options->output = "output.nii.gz";
	options->backend = getenv("BRAINCHOPC_BACKEND");
	options->crop_percent = 2.0;
	/* Not 0, which memset above would leave and which is a real adapter index.
	   -1 is "the backend chooses", i.e. the behaviour before --adapter existed. */
	options->adapter = -1;
	/* Name this executable in its own usage text rather than using a fixed
	   string, so each per-model binary describes itself. */
	const char *program = (argc > 0 && argv[0] && *argv[0]) ? argv[0] : "brainchop";
	const char *slash = strrchr(program, '/');
	if (slash) program = slash + 1;

	for (int index = 1; index < argc; index++) {
		const char *argument = argv[index];
		if (!strcmp(argument, "-h") || !strcmp(argument, "--help")) {
			bc_cli_help(stdout, program);
			return 1;
		} else if (!strcmp(argument, "-v") || !strcmp(argument, "--version")) {
			printf("%s %s\n", program, BC_VERSION);
			return 1;
		} else if (!strcmp(argument, "-o") || !strcmp(argument, "--output")) {
			if (needs_value(index, argc, argument)) return 2;
			options->output = argv[++index];
		} else if (!strcmp(argument, "--ct")) {
			options->ct = 1;
#if BC_HAS_TUNABLE_BACKEND
		} else if (!strcmp(argument, "--tune")) {
			options->tune = 1;
#endif
		} else if (!strcmp(argument, "--no-turbo")) {
			options->no_turbo = 1;
#if BC_HAS_SELECTABLE_ADAPTER
		} else if (!strcmp(argument, "-a") || !strcmp(argument, "--adapter")) {
			/*
			 * The value is consumed unconditionally rather than being probed for
			 * a leading '-', because the listing form is spelled `-a -1` and a
			 * leading-dash test would send it to the unrecognized-argument
			 * branch. parse_long is what rejects `-a input.nii.gz`, and it does
			 * so before the file is opened.
			 */
			if (needs_value(index, argc, argument)) return 2;
			long adapter;
			if (parse_long(argv[++index], &adapter) ||
					adapter < -1 || adapter > 1024) {
				fprintf(stderr, "brainchop: %s expects an adapter index, "
					"or -1 to list the available adapters\n", argument);
				return 2;
			}
			/*
			 * -1 is BOTH the internal "let the backend choose" default and the
			 * spelling the user types to get a listing, so the two cannot be
			 * told apart from the stored value afterwards. They are separated
			 * here, where the parser still knows it saw the argument: listing
			 * happens now and returns like --help does, so nothing downstream
			 * needs a second flag to carry the distinction.
			 */
			if (adapter < 0) {
				bc_backend_list_adapters(stdout);
				return 1;
			}
			options->adapter = (int)adapter;
#endif
		} else if (!strcmp(argument, "--comply")) {
			options->comply = 1;
		} else if (!strcmp(argument, "--export-classes")) {
			options->export_classes = 1;
#if BC_HAS_LEGACY_CLEANUP
		} else if (!strcmp(argument, "--legacy-cleanup")) {
			options->legacy_cleanup = 1;
#endif
#if BC_HAS_PVE
		} else if (!strcmp(argument, "--pve") ||
				!strcmp(argument, "--partial-volume-estimate")) {
			options->pve = 1;
#endif
#if BC_SUPPORTS_SAVE_CONFORM
		} else if (!strcmp(argument, "--save-conform")) {
			options->save_conform = 1;
#endif
#if BC_SUPPORTS_CROP
		} else if (!strcmp(argument, "--crop")) {
			/*
			 * Optional value. Testing whether it parses as a number is both
			 * simpler and more accurate than checking for a leading '-', which
			 * sent `--crop -5` to the unrecognized-argument branch instead of
			 * the range check below.
			 */
			options->crop = 1;
			double percent;
			if (index + 1 < argc && !parse_double(argv[index + 1], &percent)) {
				options->crop_percent = percent;
				index++;
			}
#endif
#if BC_SUPPORTS_MASK
		} else if (!strcmp(argument, "--mask")) {
			options->mask_output = "mask.nii.gz";
			if (index + 1 < argc && argv[index + 1][0] != '-')
				options->mask_output = argv[++index];
#endif
#if BC_SUPPORTS_BORDER
		} else if (!strcmp(argument, "--border")) {
			if (needs_value(index, argc, argument)) return 2;
			long border;
			if (parse_long(argv[++index], &border) || border < 0 || border > INT_MAX) {
				fprintf(stderr, "brainchop: --border expects a nonnegative integer\n");
				return 2;
			}
			options->border_mm = (int)border;
#endif
		} else if (!strcmp(argument, "-cpu")) {
			if (needs_value(index, argc, argument)) return 2;
			options->cpu_requested = 1;
			if (parse_long(argv[++index], &options->cpu_threads)) {
				fprintf(stderr, "brainchop: -cpu expects an integer\n");
				return 2;
			}
		} else if (!strcmp(argument, "-backend") || !strcmp(argument, "--backend")) {
			if (needs_value(index, argc, argument)) return 2;
			options->backend = argv[++index];
		} else if (argument[0] == '-') {
			if (rejected_option(argument)) return 2;
			fprintf(stderr, "brainchop: unrecognized argument '%s'\n", argument);
			return 2;
		} else if (!options->input) {
			options->input = argument;
		} else {
			fprintf(stderr, "brainchop: only one input is currently supported\n");
			return 2;
		}
	}
	/*
	 * A bare invocation is a request for usage. Once any argument is present a
	 * missing input is a mistake and must not exit 0 -- notably `--mask in.nii`,
	 * where the optional-value flag swallows the filename.
	 */
	if (!options->input) {
		if (argc <= 1) {
			bc_cli_help(stdout, program);
			return 1;
		}
		fprintf(stderr, "brainchop: no input image given\n");
		return 2;
	}
	if (options->crop &&
			(!(options->crop_percent >= 0.0) || !(options->crop_percent <= 100.0))) {
		fprintf(stderr, "brainchop: --crop percentile must be between 0 and 100\n");
		return 2;
	}
#if BC_HAS_LEGACY_CLEANUP
	if (options->legacy_cleanup && (options->export_classes
#if BC_HAS_PVE
			|| options->pve
#endif
			)) {
		fprintf(stderr, "brainchop: --legacy-cleanup applies only to label output\n");
		return 2;
	}
#endif
	return 0;
}
