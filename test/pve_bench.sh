#!/bin/sh
# Wall time and peak memory footprint of mindmap --pve, CPU vs Metal, on every
# *.nii.gz in BC_PVE_SCANS. Runs alternate so load drift hits both sides alike;
# still measure on an idle machine. macOS only (/usr/bin/time -l).
# Time is reported, not gated (machine noise). Exits 1 if a Metal run's peak
# footprint exceeds BC_PVE_PEAK_GB (default 4.0: Metal seg peak + host planes;
# the project's Metal arena-release trap records why it is not the CPU's 3.912).
# Usage: BC_PVE_SCANS=dir test/pve_bench.sh [rounds]   (repo root, after make models)
set -eu
exe=${BC_EXE:-src/brainchop-mindmap}
scans=${BC_PVE_SCANS:?set BC_PVE_SCANS to a directory of *.nii.gz}
rounds=${1:-3}
limit=${BC_PVE_PEAK_GB:-4.0}
over=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf '%-24s %-10s %5s %8s %9s %9s %9s %9s %9s\n' scan mode round wall_s \
	peak_GB seg_ms priors_ms fit_ms write_ms
for scan in "$scans"/*.nii.gz; do
	name=$(basename "$scan" .nii.gz)
	r=1
	while [ "$r" -le "$rounds" ]; do
		for mode in cpu-pve metal-pve; do
			case $mode in
			cpu-pve) args="-cpu 0 --pve" ;;
			metal-pve) args="-backend metal --pve" ;;
			esac
			# shellcheck disable=SC2086
			BRAINCHOPC_PROFILE=1 /usr/bin/time -l "$exe" "$scan" $args \
				-o "$tmp/out.nii.gz" >"$tmp/log" 2>&1 ||
				{ cat "$tmp/log" >&2; exit 1; }
			awk -v s="$name" -v m="$mode" -v r="$r" '
				/ real /{w=$1} /peak memory footprint/{p=$1/1e9}
				/host segment/{g=$(NF-1)} /host pve priors/{q=$(NF-1)}
				/host pve fit/{f=$(NF-1)} /host pve write/{x=$(NF-1)}
				END{printf "%-24s %-10s %5d %8.2f %9.3f %9s %9s %9s %9s\n",
					s,m,r,w,p,g,(q==""?"-":q),(f==""?"-":f),(x==""?"-":x)}' "$tmp/log"
			[ "$mode" = metal-pve ] && awk -v l="$limit" \
				'/peak memory footprint/{exit !($1/1e9 > l)}' "$tmp/log" && over=1
		done
		r=$((r + 1))
	done
done
[ "$over" = 0 ] || { echo "Metal --pve peak footprint exceeded $limit GB" >&2; exit 1; }
