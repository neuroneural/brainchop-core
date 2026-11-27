// Literal specifiers, so a host bundler (Vite, Rollup, webpack) follows each import / URL
// and emits the corresponding asset: no copying, no assetPath.
// Plain .js because staged dist/ artifacts sit beside the bundle and tsc cannot resolve them;
// modules.d.ts carries the type.
//
// WebGPU runs via the pure-JS V4 engine reading `models/<model>.bin` directly (zero -gpu.wasm!).
// WebGL2 and CPU fallbacks use Emscripten WASM modules.
export const MODEL_BINS = {
  mindgrab: () => new URL('./models/mindgrab.bin', import.meta.url).href,
  'mindmap-fast': () => new URL('./models/mindmap-fast.bin', import.meta.url).href,
  '16chan18cls': () => new URL('./models/mindmap-fast.bin', import.meta.url).href,
  mindmap: () => new URL('./models/mindmap.bin', import.meta.url).href,
  'mindmap-omni': () => new URL('./models/mindmap-omni.bin', import.meta.url).href,
  mindsnap: () => new URL('./models/mindsnap.bin', import.meta.url).href,
}

export const MODULES = {
  webgl2: {
    mindgrab: () => import('./brainchop-mindgrab-gl.js'),
    'mindmap-fast': () => import('./brainchop-mindmap-fast-gl.js'),
    '16chan18cls': () => import('./brainchop-mindmap-fast-gl.js'),
    mindmap: () => import('./brainchop-mindmap-gl.js'),
    'mindmap-omni': () => import('./brainchop-mindmap-omni-gl.js'),
    mindsnap: () => import('./brainchop-mindsnap-gl.js'),
  },
  cpu: {
    mindgrab: () => import('./brainchop-mindgrab.js'),
    'mindmap-fast': () => import('./brainchop-mindmap-fast.js'),
    '16chan18cls': () => import('./brainchop-mindmap-fast.js'),
    mindmap: () => import('./brainchop-mindmap.js'),
    'mindmap-omni': () => import('./brainchop-mindmap-omni.js'),
    mindsnap: () => import('./brainchop-mindsnap.js'),
  },
}
