# Browser WASM CPU Profiling

This directory holds local profiler captures (ignored by Git via `/profiles/*`)
and recipes for generating symbol-mapped WebAssembly builds when profiling
multi-threaded CPU fallback performance in a browser.

## Build a symbol-mapped WASM module

Build into a separate directory (`WASM_DIR`) so the symbol-mapped output does
not overwrite `dist/wasm`:

```sh
make -C src test-wasm
make -C src wasm-model16chan18cls \
  WASM_DIR=/tmp/brainchopc-wasm-symbols WASM_CPU_SYMBOL_MAP=1
```

## Run against an installed browser

Always benchmark against a release browser (not an unoptimized test/debug
browser build):

```sh
BC_MT_MODES=release \
BC_FIREFOX_PATH=/Applications/Firefox.app/Contents/MacOS/firefox \
node test/wasm_mt_run.mjs \
  /tmp/brainchopc-wasm-symbols/brainchop-16chan18cls.js \
  /tmp/bc-w-t12mm.nii /tmp/bc-w-m16n.nii
```

Note: `WASM_RELAXED_SIMD=1` is available as an opt-in build flag (~1.4× faster
CPU WASM in Node/Chromium) and is disabled by default to preserve portable
bit-exactness across engines.
