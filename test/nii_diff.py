#!/usr/bin/env python3
"""Count differing voxels between two NIfTI files.

usage: nii_diff.py A B [MAX_DIFFERING]

Reads the voxel datatype from the header rather than taking it as an argument,
so the same helper works for the int16 MindGrab output, the uint8 label maps and
the uint8 masks. Exits 1 if the headers disagree, the volumes differ in size, or
the differing-voxel count exceeds MAX_DIFFERING (default 0).
"""

import array
import gzip
import struct
import sys

# NIfTI-1 datatype code -> (array typecode, bytes per voxel)
TYPECODES = {
    2: ("B", 1),    # DT_UINT8
    4: ("h", 2),    # DT_INT16
    8: ("i", 4),    # DT_INT32
    16: ("f", 4),   # DT_FLOAT32
    64: ("d", 8),   # DT_FLOAT64
    256: ("b", 1),  # DT_INT8
    512: ("H", 2),  # DT_UINT16
    768: ("I", 4),  # DT_UINT32
}

HEADER_BYTES = 352
DESCRIP_START = 148
DESCRIP_END = 228


def read(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return handle.read()


def header_matches(a_blob, b_blob):
    """Compare all 352 NIfTI-1 header bytes except descrip (bytes 148..228),
    which records per-run provenance (version, model, backend, precision mode)."""
    return (
        a_blob[:DESCRIP_START] == b_blob[:DESCRIP_START]
        and a_blob[DESCRIP_END:HEADER_BYTES] == b_blob[DESCRIP_END:HEADER_BYTES]
    )


def equal_ignoring_descrip(a_blob, b_blob):
    """Compare two uncompressed NIfTI-1 blobs ignoring only descrip (148..228)."""
    return (
        len(a_blob) == len(b_blob)
        and header_matches(a_blob, b_blob)
        and a_blob[HEADER_BYTES:] == b_blob[HEADER_BYTES:]
    )


def descrip(blob):
    """Decode the NUL-terminated ASCII descrip field from a NIfTI-1 blob."""
    return blob[DESCRIP_START:DESCRIP_END].split(b"\0", 1)[0].decode("ascii", "replace")


def voxels(blob, path):
    datatype = struct.unpack_from("<h", blob, 70)[0]
    if datatype not in TYPECODES:
        sys.exit(f"nii_diff: {path}: unsupported datatype {datatype}")
    code, width = TYPECODES[datatype]
    body = blob[HEADER_BYTES:]
    if len(body) % width:
        sys.exit(f"nii_diff: {path}: body is not a whole number of voxels")
    values = array.array(code)
    values.frombytes(body)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main():
    if not 3 <= len(sys.argv) <= 4:
        sys.exit(__doc__)
    a_path, b_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    a_blob, b_blob = read(a_path), read(b_path)
    if not header_matches(a_blob, b_blob):
        sys.exit(f"nii_diff: headers differ between {a_path} and {b_path}")

    a, b = voxels(a_blob, a_path), voxels(b_blob, b_path)
    if len(a) != len(b):
        sys.exit(f"nii_diff: {len(a)} voxels vs {len(b)} voxels")

    differing = sum(x != y for x, y in zip(a, b))
    print(f"{differing} of {len(a)} voxels differ (limit {limit})")
    if differing > limit:
        sys.exit(1)


if __name__ == "__main__":
    main()
