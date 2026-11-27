// Each model's own colormap.json, bundled by esbuild. Plain .js for the same reason as
// modules.js: the files sit outside tsc's rootDir; colormaps.d.ts carries the type.
import mindgrab from '../../src/models/mindgrab/colormap.json'
import model16 from '../../src/models/model16chan18cls/colormap.json'
import mindmap from '../../src/models/mindmap/colormap.json'
import mindmapOmni from '../../src/models/mindmap_omni/colormap.json'
import mindsnap from '../../src/models/mindsnap/colormap.json'

export const COLORMAPS = {
  mindgrab,
  'mindmap-fast': model16,
  '16chan18cls': model16,
  mindmap,
  'mindmap-omni': mindmapOmni,
  mindsnap,
}
