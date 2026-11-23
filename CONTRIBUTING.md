# Contributing to BrainChop Core

Contributions, bug reports, and performance benchmarks across hardware platforms are welcome!

## Development Setup

```bash
git clone https://github.com/neuroneural/brainchop-core.git
cd brainchop-core

# Build all four model binaries + the brainchop dispatcher
make -C src -j4
```

See [docs/building.md](docs/building.md) and [docs/testing-and-wasm.md](docs/testing-and-wasm.md) for platform-specific prerequisites (macOS Metal, Linux GCC/Clang/CUDA/HIP, Windows MSYS2 UCRT64, and Emscripten WASM).

## Running Tests

Before opening a pull request, run the relevant verification suites:

```bash
# Native CPU bit-exact parity, CLI guards, cleanup & reslice suites
make -C src test

# Apple Silicon Metal exact + Turbo + PVE tests (macOS only)
make -C src test-metal

# Python packaging, adaptive ABI, CLI, and Python API unit tests
python3 -m unittest discover -s test -p 'test_pypi*.py'
```

## Coding Guidelines

- **Zero runtime dependencies:** Keep the C runtime dependent only on standard OS libraries and `zlib`, and keep `packaging/` (`brainchop` and `brainchop_cli`) pure Python standard library (`numpy` optional only).
- **Exactness & parity:** Changes to preprocessing (`conform.c`), cleanup (`bwlabel.c`), or PVE (`cat_lite.c`) must maintain parity with the browser implementation and pass `make -C src test`.
- **No unverified claims:** Include exact hardware details, commands run, and observed Dice / voxel-diff results in pull requests touching compute kernels or models.
- **Keep private paths out of commits:** Never commit personal home paths, internal cluster hostnames, hardware UUIDs, or credentials.
