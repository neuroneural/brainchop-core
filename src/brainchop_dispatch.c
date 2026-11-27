/*
 * brainchop_dispatch.c - Tiny zero-dependency CLI dispatcher for standalone
 * native installations (Homebrew, macOS .pkg, Linux/Windows release archives).
 *
 * Locates the sibling `brainchop-<model>` executable beside this binary and
 * forwards all remaining arguments directly to it. Python installations use
 * `brainchop_cli` for the `brainchop` entry point (adding multi-file batching
 * and legacy CLI compatibility flags); native archives use this C binary so
 * `brainchop [-m MODEL] ...` works with zero Python or runtime dependencies.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "version.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define BC_PATH_MAX 32768
#define BC_EXE_EXT ".exe"
#else
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#define BC_PATH_MAX PATH_MAX
#define BC_EXE_EXT ""
#endif

typedef struct {
	const char *name;
	const char *description;
} bc_model_entry;

static const bc_model_entry BC_MODELS[] = {
	{ "mindgrab",     "Skull stripping in any modality" },
	{ "mindmap",      "18-class cortical and subcortical segmentation (default)" },
	{ "mindmap-omni", "Contrast-agnostic 18-class segmentation (T2w, FLAIR, T1w, ULF, PET)" },
	{ "mindsnap",     "104-class Desikan-Killiany cortical and subcortical parcellation" },
	{ "mindmap-fast", "Fast lightweight 18-class segmentation" },
};

static const size_t BC_MODEL_COUNT = sizeof(BC_MODELS) / sizeof(BC_MODELS[0]);

static void print_list(FILE *out) {
	for (size_t i = 0; i < BC_MODEL_COUNT; i++)
		fprintf(out, "  %-14s %s\n", BC_MODELS[i].name, BC_MODELS[i].description);
}

static void print_help(FILE *out, const char *prog) {
	fprintf(out,
		"usage: %s [-h] [-v] [-l] [-m MODEL | --skull-strip] [MODEL_OPTIONS ...] INPUT\n\n"
		"BrainChop %s: native brain segmentation and skull stripping\n\n"
		"Dispatcher options:\n"
		"  -m, --model MODEL   select model (default: mindmap)\n"
		"      --skull-strip   shorthand for --model mindgrab\n"
		"  -l, --list          list embedded models\n"
		"  -v, --version       show version and exit\n"
		"  -h, --help          show this help (use `-m MODEL --help` for model options)\n\n"
		"Models:\n",
		prog, BC_VERSION);
	print_list(out);
	fprintf(out,
		"\nExamples:\n"
		"  %s -o labels.nii.gz input.nii.gz\n"
		"  %s -m mindmap --pve -o brain.nii.gz input.nii.gz\n"
		"  %s -m mindsnap -o parcels.nii.gz input.nii.gz\n"
		"  %s --skull-strip --mask brain_mask.nii.gz -o stripped.nii.gz input.nii.gz\n"
		"\nAll other options are forwarded verbatim to `brainchop-<model>`.\n",
		prog, prog, prog, prog);
}

static const char *resolve_model(const char *raw) {
	if (!raw || !raw[0]) return NULL;
	if (strcmp(raw, "16chan18cls") == 0 || strcmp(raw, "model16chan18cls") == 0)
		return "mindmap-fast";
	if (strcmp(raw, "mindmap_omni") == 0)
		return "mindmap-omni";
	for (size_t i = 0; i < BC_MODEL_COUNT; i++) {
		if (strcmp(raw, BC_MODELS[i].name) == 0)
			return BC_MODELS[i].name;
	}
	return NULL;
}

static const char *legacy_hint(const char *raw) {
	if (!raw) return NULL;
	if (strcmp(raw, "subcortical") == 0 || strcmp(raw, "subcortical_mini") == 0 ||
	    strcmp(raw, "tissue_fast") == 0 || strcmp(raw, "tissue") == 0)
		return "Try `-m mindmap` (18-class, or `--pve` for GM/WM/CSF) or `-m mindmap-fast`.";
	if (strcmp(raw, "DKatlas") == 0 || strcmp(raw, "dkatlas") == 0 ||
	    strcmp(raw, "aparc") == 0 || strcmp(raw, "parcellation") == 0)
		return "Try `-m mindsnap` (104-class Desikan-Killiany parcellation).";
	return NULL;
}

/* Write the directory containing this executable (no trailing slash) into out. */
static int find_self_dir(const char *argv0, char *out, size_t out_size) {
	char raw[BC_PATH_MAX];
	raw[0] = '\0';

#if defined(_WIN32)
	DWORD len = GetModuleFileNameA(NULL, raw, (DWORD)sizeof(raw));
	if (len == 0 || len >= sizeof(raw)) return -1;
	raw[len] = '\0';
#elif defined(__APPLE__)
	uint32_t sz = (uint32_t)sizeof(raw);
	if (_NSGetExecutablePath(raw, &sz) != 0) return -1;
	char resolved[BC_PATH_MAX];
	if (realpath(raw, resolved)) {
		if (strlen(resolved) >= sizeof(raw)) return -1;
		strcpy(raw, resolved);
	}
#elif defined(__linux__)
	ssize_t len = readlink("/proc/self/exe", raw, sizeof(raw) - 1);
	if (len <= 0 || (size_t)len >= sizeof(raw)) return -1;
	raw[len] = '\0';
#else
	if (!argv0 || !argv0[0] || strlen(argv0) >= sizeof(raw)) return -1;
	strcpy(raw, argv0);
#endif
	(void)argv0;

	char *sep = strrchr(raw, '/');
#if defined(_WIN32)
	char *bsep = strrchr(raw, '\\');
	if (bsep && (!sep || bsep > sep)) sep = bsep;
#endif
	if (!sep) return -1;
	size_t dir_len = (size_t)(sep - raw);
	if (dir_len == 0) dir_len = 1; /* root "/" */
	if (dir_len + 1 > out_size) return -1;
	memcpy(out, raw, dir_len);
	out[dir_len] = '\0';
	return 0;
}

#if defined(_WIN32)
static int append_str(char **buf, size_t *len, size_t *cap, const char *s, size_t n) {
	if (*len + n + 1 > *cap) {
		size_t next = (*cap ? *cap : 256);
		while (next < *len + n + 1) next *= 2;
		char *p = (char *)realloc(*buf, next);
		if (!p) return -1;
		*buf = p;
		*cap = next;
	}
	memcpy(*buf + *len, s, n);
	*len += n;
	(*buf)[*len] = '\0';
	return 0;
}

/* Windows CommandLineToArgvW-compatible quoting for CreateProcessA. */
static int append_win_arg(char **buf, size_t *len, size_t *cap, const char *arg) {
	int needs_quotes = (arg[0] == '\0');
	for (const char *p = arg; *p && !needs_quotes; p++) {
		if (*p == ' ' || *p == '\t' || *p == '"') needs_quotes = 1;
	}
	if (!needs_quotes) return append_str(buf, len, cap, arg, strlen(arg));
	if (append_str(buf, len, cap, "\"", 1) != 0) return -1;
	size_t backslashes = 0;
	for (const char *p = arg; *p; p++) {
		if (*p == '\\') {
			backslashes++;
		} else if (*p == '"') {
			for (size_t i = 0; i < backslashes * 2 + 1; i++)
				if (append_str(buf, len, cap, "\\", 1) != 0) return -1;
			if (append_str(buf, len, cap, "\"", 1) != 0) return -1;
			backslashes = 0;
		} else {
			for (size_t i = 0; i < backslashes; i++)
				if (append_str(buf, len, cap, "\\", 1) != 0) return -1;
			backslashes = 0;
			if (append_str(buf, len, cap, p, 1) != 0) return -1;
		}
	}
	for (size_t i = 0; i < backslashes * 2; i++)
		if (append_str(buf, len, cap, "\\", 1) != 0) return -1;
	return append_str(buf, len, cap, "\"", 1);
}

static int run_child(const char *exe_path, char **child_argv) {
	char *cmdline = NULL;
	size_t len = 0, cap = 0;
	for (int i = 0; child_argv[i]; i++) {
		if (i > 0 && append_str(&cmdline, &len, &cap, " ", 1) != 0) {
			free(cmdline);
			return 1;
		}
		if (append_win_arg(&cmdline, &len, &cap, child_argv[i]) != 0) {
			free(cmdline);
			return 1;
		}
	}
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	ZeroMemory(&pi, sizeof(pi));
	si.cb = sizeof(si);
	DWORD attr = exe_path ? GetFileAttributesA(exe_path) : INVALID_FILE_ATTRIBUTES;
	const char *app = (attr != INVALID_FILE_ATTRIBUTES) ? exe_path : NULL;
	if (!CreateProcessA(app, cmdline, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
		DWORD err = GetLastError();
		fprintf(stderr, "brainchop: failed to launch %s (Windows error %lu)\n",
			app ? app : child_argv[0], (unsigned long)err);
		free(cmdline);
		return 127;
	}
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD exit_code = 1;
	GetExitCodeProcess(pi.hProcess, &exit_code);
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	free(cmdline);
	return (int)exit_code;
}
#else
static int run_child(const char *exe_path, char **child_argv) {
	if (exe_path && access(exe_path, X_OK) == 0) {
		execv(exe_path, child_argv);
		fprintf(stderr, "brainchop: failed to execute %s: %s\n",
			exe_path, strerror(errno));
		return 127;
	}
	execvp(child_argv[0], child_argv);
	fprintf(stderr, "brainchop: could not find %s (tried %s and PATH): %s\n",
		child_argv[0], exe_path ? exe_path : "<unknown>", strerror(errno));
	return 127;
}
#endif

int main(int argc, char **argv) {
	const char *prog = (argc > 0 && argv[0] && argv[0][0]) ? argv[0] : "brainchop";
	const char *requested_model = NULL;
	int explicit_model = 0;
	int saw_help = 0;
	int saw_version = 0;
	int saw_list = 0;

	char **forward = (char **)calloc((size_t)argc + 2, sizeof(char *));
	if (!forward) {
		fprintf(stderr, "brainchop: out of memory\n");
		return 1;
	}
	int fwd_n = 1; /* slot 0 reserved for target executable name */
	int stop_opts = 0;

	for (int i = 1; i < argc; i++) {
		const char *arg = argv[i];
		if (!stop_opts && strcmp(arg, "--") == 0) {
			stop_opts = 1;
			forward[fwd_n++] = argv[i];
			continue;
		}
		if (!stop_opts) {
			if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
				if (i + 1 >= argc) {
					fprintf(stderr, "brainchop: %s requires a model name\n", arg);
					free(forward);
					return 2;
				}
				requested_model = argv[++i];
				explicit_model = 1;
				continue;
			}
			if (strncmp(arg, "--model=", 8) == 0) {
				requested_model = arg + 8;
				explicit_model = 1;
				continue;
			}
			if (strcmp(arg, "--skull-strip") == 0) {
				requested_model = "mindgrab";
				explicit_model = 1;
				continue;
			}
			if (strcmp(arg, "-l") == 0 || strcmp(arg, "--list") == 0) {
				saw_list = 1;
				continue;
			}
			if (strcmp(arg, "-v") == 0 || strcmp(arg, "--version") == 0) {
				saw_version = 1;
				continue;
			}
			if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
				saw_help = 1;
				forward[fwd_n++] = argv[i];
				continue;
			}
		}
		forward[fwd_n++] = argv[i];
	}

	if (saw_version && !explicit_model) {
		printf("brainchop %s\n", BC_VERSION);
		free(forward);
		return 0;
	}
	if (saw_list) {
		print_list(stdout);
		free(forward);
		return 0;
	}
	if (argc == 1 || (saw_help && !explicit_model)) {
		print_help(stdout, prog);
		free(forward);
		return 0;
	}

	const char *model_name = explicit_model ? requested_model : "mindmap";
	const char *canonical = resolve_model(model_name);
	if (!canonical) {
		fprintf(stderr,
			"brainchop: unknown model '%s'. Available models: mindgrab, mindmap, mindmap-omni, mindsnap, mindmap-fast.\n",
			model_name);
		const char *hint = legacy_hint(model_name);
		if (hint) fprintf(stderr, "brainchop: %s\n", hint);
		free(forward);
		return 2;
	}
	static char ver_flag[] = "--version";
	if (saw_version)
		forward[fwd_n++] = ver_flag;

	char exe_name[64];
	snprintf(exe_name, sizeof(exe_name), "brainchop-%s%s", canonical, BC_EXE_EXT);
	forward[0] = exe_name;
	forward[fwd_n] = NULL;

	char dir[BC_PATH_MAX];
	char full_path[BC_PATH_MAX];
	const char *target = NULL;
	if (find_self_dir(prog, dir, sizeof(dir)) == 0) {
#if defined(_WIN32)
		int n = snprintf(full_path, sizeof(full_path), "%s\\%s", dir, exe_name);
#else
		int n = snprintf(full_path, sizeof(full_path), "%s/%s", dir, exe_name);
#endif
		if (n > 0 && (size_t)n < sizeof(full_path)) target = full_path;
	}

	int status = run_child(target, forward);
	free(forward);
	return status;
}
