#ifndef BC_VERSION_H
#define BC_VERSION_H

/*
 * THE version string. Everything that reports one derives it from here:
 * `bc_cli_help` prints it, `run_model` prints it once per run (so a browser
 * console shows it too), `js/scripts/copy-wasm.mjs` refuses to build if
 * `js/package.json` has drifted from it, and `scripts/package_macos.sh` takes
 * its default installer version from it.
 *
 * 0.1 says pre-release: the CLI surface is not promised to be stable. The third
 * field is the build date as YYYYMMDD rather than a patch count, because
 * releases here are cut when something lands rather than on a schedule, and a
 * date answers "is my copy older than the fix" without a changelog. It is also
 * a valid semver patch number, which is what lets js/package.json carry the
 * identical string.
 *
 * Bump it by hand. Deriving it from `git describe` or the build clock was the
 * obvious alternative and is worse: it would make two builds of the same source
 * differ, and this project's whole test strategy is byte-identical output from
 * byte-identical input.
 */
#define BC_VERSION "1.1.3"

#endif
