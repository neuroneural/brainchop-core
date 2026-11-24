#!/usr/bin/env python3
"""Check that MindGrab preserves scaled integer NIfTI intensities."""

from array import array
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from nii_diff import HEADER_BYTES, read, voxels


def main(executable, fixture):
    source = read(fixture)
    original = voxels(source, fixture)
    minimum = min(original)
    background = -32768
    slope, intercept = 512, 512 * -background
    stored = array("h", (v - minimum + background for v in original))
    little_endian = stored if sys.byteorder == "little" else array("h", stored)
    if sys.byteorder != "little":
        little_endian.byteswap()
    header = bytearray(source[:HEADER_BYTES])
    struct.pack_into("<2f", header, 112, slope, intercept)

    with TemporaryDirectory(prefix="brainchop-mindgrab-scaling-") as temp:
        input_path = Path(temp) / "scaled.nii"
        output_path = Path(temp) / "stripped.nii"
        mask_path = Path(temp) / "mask.nii"
        input_path.write_bytes(header + little_endian.tobytes())
        subprocess.run(
            [executable, "-cpu", "0", "--mask", str(mask_path), "-o", str(output_path), str(input_path)],
            check=True,
        )
        mask = voxels(read(str(mask_path)), mask_path)
        result = read(str(output_path))
        assert set(mask) == {0, 1}, "expected brain and background voxels"
        assert any(keep and raw * slope + intercept > 32767 for raw, keep in zip(stored, mask))
        assert struct.unpack_from("<2h", result, 70) == (4, 16), "output datatype changed"
        assert struct.unpack_from("<2f", result, 112) == (slope, intercept), "output scaling changed"
        output = voxels(result, output_path)
        assert len(output) == len(stored) == len(mask), "output size changed"
        assert all(value == (raw if keep else background)
                   for value, raw, keep in zip(output, stored, mask)), "voxel values changed"
    print("MindGrab scaling: voxel values and scaling preserved")


if __name__ == "__main__":
    main(*sys.argv[1:])
