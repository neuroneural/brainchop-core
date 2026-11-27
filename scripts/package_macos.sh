#!/bin/sh
# Build a signed macOS installer package containing one executable per model.
#
#   package_macos.sh <version>
#
# Environment:
#   MACOS_SIGN_IDENTITY       "Developer ID Application: ..." or its SHA-1, or
#                             "-" for an ad-hoc signature. Signs the executables.
#   MACOS_INSTALLER_IDENTITY  "Developer ID Installer: ..." or its SHA-1. Signs
#                             the .pkg itself. This is a DIFFERENT certificate
#                             from the Application one, and a package signed
#                             with the Application certificate is rejected.
#                             Leave unset for an unsigned package, which is for
#                             local testing only and cannot be notarised.
#
# The package installs to /usr/local/bin, which is on the default PATH via
# /etc/paths, so the tools work immediately without the user editing a shell
# profile. That convenience is the whole reason to prefer a package over a disk
# image for command-line tools.
#
# Each executable is a single self-contained file: the weights are embedded by
# payload.S and threading is pthreads from libSystem, so there is no lib/
# directory to stage, no install_name_tool rewriting and no second signature.
# Everything it links against ships with macOS.
set -eu

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
	echo "package_macos.sh requires Apple Silicon macOS" >&2
	exit 2
fi

bc_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bc_src="$bc_root/src"
# Default from src/version.h, which owns the string; an explicit argument still
# wins, so a one-off build can be labelled without editing the header.
bc_default_version=$(sed -n 's/^#define BC_VERSION "\(.*\)"/\1/p' "$bc_root/src/version.h")
bc_version=${1:-${bc_default_version:-1.1.2}}
bc_identity=${MACOS_SIGN_IDENTITY:--}
bc_installer_identity=${MACOS_INSTALLER_IDENTITY:-}
bc_target=${MACOSX_DEPLOYMENT_TARGET:-13.2}
bc_identifier=org.brainchop.brainchop
bc_dist="$bc_root/dist"
bc_pkg="$bc_dist/brainchop-$bc_version-macos-arm64.pkg"

case "$bc_version" in
	*[!A-Za-z0-9._-]*|'')
		echo "package_macos.sh: VERSION may contain only letters, digits, dot, underscore, and hyphen" >&2
		exit 2
		;;
esac
case "$bc_target" in
	*[!0-9.]*|'')
		echo "package_macos.sh: deployment target must contain only digits and dots" >&2
		exit 2
		;;
esac

for bc_tool in codesign pkgbuild productbuild pkgutil otool xattr; do
	command -v "$bc_tool" >/dev/null 2>&1 || {
		echo "package_macos.sh: missing required tool: $bc_tool" >&2
		exit 2
	}
done

if [ "$bc_identity" != - ]; then
	security find-identity -v -p codesigning | grep -F -- "$bc_identity" >/dev/null 2>&1 || {
		echo "package_macos.sh: signing identity not found in the current Keychain: $bc_identity" >&2
		exit 2
	}
fi
# Installer certificates are not codesigning identities, so they are absent from
# `find-identity -p codesigning`. Look at the full list instead; getting this
# wrong is easy and productbuild's own error is unhelpful.
if [ -n "$bc_installer_identity" ]; then
	security find-identity -v | grep -F -- "$bc_installer_identity" >/dev/null 2>&1 || {
		echo "package_macos.sh: installer identity not found in the current Keychain: $bc_installer_identity" >&2
		echo "  This must be a 'Developer ID Installer' certificate, which is separate" >&2
		echo "  from the 'Developer ID Application' certificate used above." >&2
		exit 2
	}
fi

# Nothing to build before the models: threading is pthreads from libSystem and
# zlib is the system one, so there is no dependency to stage. The minos
# assertion below is what would catch a regression in that.
make -C "$bc_src" clean >/dev/null
env -u LIBRARY_PATH -u CPATH -u LDFLAGS -u CPPFLAGS -u CFLAGS \
	MACOSX_DEPLOYMENT_TARGET="$bc_target" make -C "$bc_src" models dispatcher ZSTD=0

# Taken from the Makefile rather than written out again here, so the package
# cannot drift from the set of executables that was actually built.
bc_exes=$(make -C "$bc_src" --no-print-directory print-release-exes)
[ -n "$bc_exes" ] || { echo "package_macos.sh: no release executables" >&2; exit 1; }

bc_payload=$(mktemp -d "${TMPDIR:-/tmp}/brainchopc-pkg.XXXXXX")
trap 'rm -rf "$bc_payload"' EXIT HUP INT TERM
mkdir -p "$bc_payload/root/usr/local/bin"

for bc_exe in $bc_exes; do
	[ -f "$bc_src/$bc_exe" ] || {
		echo "package_macos.sh: $bc_exe was not built" >&2
		exit 1
	}
	# A non-system dependency here would mean the installed tool breaks on a
	# machine that lacks it -- exactly what embedding the weights is meant to
	# prevent. Verify rather than assume.
	bc_foreign=$(otool -L "$bc_src/$bc_exe" | tail -n +2 | awk '{ print $1 }' |
		grep -v '^/usr/lib/' | grep -v '^/System/' || true)
	if [ -n "$bc_foreign" ]; then
		echo "package_macos.sh: $bc_exe has non-system dependencies:" >&2
		echo "$bc_foreign" | sed 's/^/  /' >&2
		exit 1
	fi
	bc_minos=$(otool -l "$bc_src/$bc_exe" | awk '$1 == "minos" { print $2; exit }')
	[ "$bc_minos" = "$bc_target" ] || {
		echo "package_macos.sh: $bc_exe targets macOS $bc_minos, expected $bc_target" >&2
		exit 1
	}

	cp "$bc_src/$bc_exe" "$bc_payload/root/usr/local/bin/$bc_exe"
	chmod 755 "$bc_payload/root/usr/local/bin/$bc_exe"
	xattr -cr "$bc_payload/root/usr/local/bin/$bc_exe"
	# Hardened runtime and a secure timestamp are both required for notarisation.
	if [ "$bc_identity" = - ]; then
		codesign --force --sign - "$bc_payload/root/usr/local/bin/$bc_exe"
	else
		codesign --force --options runtime --timestamp \
			--sign "$bc_identity" "$bc_payload/root/usr/local/bin/$bc_exe"
	fi
	codesign --verify --strict --verbose=2 "$bc_payload/root/usr/local/bin/$bc_exe"
done

mkdir -p "$bc_dist"
pkgbuild --root "$bc_payload/root" --identifier "$bc_identifier" \
	--version "$bc_version" --install-location / \
	"$bc_payload/component.pkg"

if [ -n "$bc_installer_identity" ]; then
	productbuild --package "$bc_payload/component.pkg" \
		--sign "$bc_installer_identity" "$bc_pkg"
else
	echo "package_macos.sh: MACOS_INSTALLER_IDENTITY is unset, so the package will" >&2
	echo "  be unsigned. It cannot be notarised and Gatekeeper will refuse it on" >&2
	echo "  another machine. Local testing only." >&2
	productbuild --package "$bc_payload/component.pkg" "$bc_pkg"
fi

echo
echo "package_macos.sh: wrote $bc_pkg"
pkgutil --check-signature "$bc_pkg" 2>&1 | sed 's/^/  /' || true
echo "  installs to /usr/local/bin:"
for bc_exe in $bc_exes; do echo "    $bc_exe"; done
