import assert from 'node:assert/strict'
import {existsSync, readFileSync, writeFileSync, mkdtempSync, rmSync} from 'node:fs'
import {tmpdir} from 'node:os'
import {join, resolve} from 'node:path'
import {pathToFileURL} from 'node:url'
import {createRequire} from 'node:module'
import {spawnSync} from 'node:child_process'

const [runner, tree] = process.argv.slice(2).map(x => resolve(x))
if (!runner || !tree) throw Error('usage: label_resample_check.mjs RUNNER BROWSER_TREE')

// Execute the actual upstream browser reslicer (modular `reslice-volume.js` in
// modern brainchop-test, with fallback to inline `main.js` for older checkouts).
let runBrowserOracle
const resliceModPath = join(tree, 'reslice-volume.js')
if (existsSync(resliceModPath)) {
  const { resliceVolumeData } = await import(pathToFileURL(resliceModPath).href)
  runBrowserOracle = (a, dims, b, targetDims, labels) =>
    resliceVolumeData({
      srcImg: labels,
      srcDims: dims,
      srcAffine: a,
      dstDims: targetDims,
      dstAffine: b,
      isLabel: true,
    }).img
} else {
  const require = createRequire(join(tree, 'package.json'))
  const {mat4} = require('gl-matrix')
  const text = readFileSync(join(tree, 'main.js'), 'utf8').replace(/\r\n/g, '\n')
  const start = text.indexOf('function resliceLabelsToNative(seg) {')
  assert(start >= 0)
  const end = text.indexOf('return outNV;\n  }', start)
  assert(end > start)
  const source = text.slice(start, end + 'return outNV;\n  }'.length)
  const makeOracle = new Function('nativeInputNV', 'mat4', 'cloneVolume', source + '\nreturn resliceLabelsToNative;')
  const hdr = (affine, shape) => ({affine, dims: [3, ...shape], intent_code: 1002})
  runBrowserOracle = (a, dims, b, targetDims, labels) =>
    makeOracle({hdr: hdr(b, targetDims)}, mat4, x => ({hdr: structuredClone(x.hdr)}))({hdr: hdr(a, dims), img: labels}).img
}

const identity = () => [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
const dir = mkdtempSync(join(tmpdir(), 'bc-label-vote-'))
let cases = 0, seed = 19237
const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2**32
function check(name, a, b, dims = [9,7,5], targetDims = [7,6,4], invalid = false, values, known) {
  const labels = values ? Float32Array.from(values) : Float32Array.from({length: dims.reduce((x,y)=>x*y)}, () => Math.floor(random()*18))
  const input = Buffer.alloc(6*4 + 32*8 + labels.byteLength)
  ;[...dims, ...targetDims].forEach((v,i) => input.writeUInt32LE(v, i*4))
  ;[...a.flat(), ...b.flat()].forEach((v,i) => input.writeDoubleLE(v, 24+i*8))
  Buffer.from(labels.buffer).copy(input, 280)
  const path = join(dir, 'in'), out = join(dir, 'out')
  writeFileSync(path, input)
  const got = spawnSync(runner, [path, out], {encoding:'utf8'})
  if (invalid) assert.notEqual(got.status, 0, name)
  else {
    assert.equal(got.status, 0, `${name}: ${got.stderr}`)
    const expected = runBrowserOracle(a, dims, b, targetDims, labels)
    if (known) assert.deepEqual(Array.from(expected), known, `${name}: independent expectation`)
    const raw = readFileSync(out)
    const actual = new Float32Array(raw.buffer, raw.byteOffset, raw.length/4)
    assert.equal(actual.length, expected.length)
    for (let i=0; i<expected.length; i++) assert.equal(actual[i], expected[i], `${name} voxel ${i}`)
  }
  cases++
}
try {
  // 1 mm / thin-slice (L <= 1.5001): streaming 2x2x2 fixed-point trilinear one-hot voting
  // with deterministic NN tie-break.
  const halfShift = identity(); halfShift[0][3] = 0.5
  check('1mm half-voxel shift ties break to NN via floor(x+0.5)', identity(), halfShift, [2,1,1], [1,1,1], false, [7,2], [2])
  const edge = identity(); edge[0][3] = -0.5
  check('1mm negative half rounds to 0 (in-bounds)', identity(), edge, [1,1,1], [1,1,1], false, [7], [7])
  const negativeHalf = identity(); negativeHalf[0][3] = -0.25
  check('negative quarter rounds to 0', identity(), negativeHalf, [1,1,1], [1,1,1], false, [7], [7])

  // 2D/3D trilinear voting beats an isolated corner even when it sits at the NN position:
  // At (0.6, 0.6, 0.6), NN corner (1,1,1)=9 has weight 0.6^3 = 0.216, while the 7 other
  // corners all hold label 3 with combined weight 1 - 0.216 = 0.784 -> 3 MUST win!
  const triPt = identity(); triPt[0][3] = 0.6; triPt[1][3] = 0.6; triPt[2][3] = 0.6
  check('3D trilinear beats isolated NN corner (0.784 vs 0.216)', identity(), triPt, [2,2,2], [1,1,1], false, [3,3,3,3,3,3,3,9], [3])
  // Multi-class junction (4 labels meeting at (0.6, 0.6, 0.4)):
  // z=0 slice (w=0.6): [1,2; 2,4], z=1 slice (w=0.4): [1,2; 2,4]
  // Label 2 appears twice at (1,0,*) and (0,1,*): weight 0.6*0.4 + 0.4*0.6 = 0.48 > label 4 (0.36) > label 1 (0.16)
  const triJunction = identity(); triJunction[0][3] = 0.6; triJunction[1][3] = 0.6; triJunction[2][3] = 0.4
  check('3D trilinear multi-class junction sums disjoint corners of same label', identity(), triJunction, [2,2,2], [1,1,1], false, [1,2,2,4, 1,2,2,4], [2])
  // Exact 3D midpoint (0.5, 0.5, 0.5) tie-break: 4 corners=5 vs 4 corners=8, NN=(1,1,1)=8 -> 8 wins
  const triMid = identity(); triMid[0][3] = 0.5; triMid[1][3] = 0.5; triMid[2][3] = 0.5
  check('3D midpoint 4-vs-4 tie breaks to NN corner (1,1,1)', identity(), triMid, [2,2,2], [1,1,1], false, [5,5,5,5, 8,8,8,8], [8])
  // Empty all-zero source (bbox early-out path)
  check('empty all-zero source', identity(), triPt, [4,4,4], [3,3,3], false, new Array(64).fill(0), new Array(27).fill(0))
  // Border-touching single voxel at corner (0,0,0) and (2,2,2) exercises OOB sampling inside bbox
  const borderVals = new Array(27).fill(0); borderVals[0] = 6; borderVals[26] = 11
  const borderShift = identity(); borderShift[0][3] = -0.2; borderShift[1][3] = -0.2; borderShift[2][3] = -0.2
  check('border-touching corners with negative shift', identity(), borderShift, [3,3,3], [3,3,3], false, borderVals)

  // 2 mm even-factor grid (L = 2.0 -> n = 4 at +/-{1/8, 3/8}): {C-1, C, C, C+1} at integer C
  const thick2 = identity(); thick2[0][0] = 2; thick2[0][3] = 1
  check('2mm integer center gives center double weight (zero half-voxel shift)', identity(), thick2, [3,1,1], [1,1,1], false, [7,2,9], [2])
  check('2mm center breaks 2-2 tie against flanking slices', identity(), thick2, [3,1,1], [1,1,1], false, [7,2,7], [2])
  check('2mm left-right reflection symmetry (left pair)', identity(), thick2, [3,1,1], [1,1,1], false, [7,7,2], [7])
  check('2mm left-right reflection symmetry (right pair)', identity(), thick2, [3,1,1], [1,1,1], false, [2,7,7], [7])
  const thick2Neg = identity(); thick2Neg[0][0] = -2; thick2Neg[0][3] = 1
  check('2mm negative step preserves center-weighted zero shift', identity(), thick2Neg, [3,1,1], [1,1,1], false, [7,2,9], [2])

  // 3 mm grid (L = 3.0 -> n = 4 at +/-{1/8, 3/8}):
  //   integer center C -> {C-1, C, C, C+1}; half-integer center C+0.5 -> {C-1, C, C+1, C+2} (zero +0.5 shift!)
  const thick3 = identity(); thick3[0][0] = 3; thick3[0][3] = 1
  check('3mm integer center includes center slice with double weight', identity(), thick3, [3,1,1], [1,1,1], false, [7,2,9], [2])
  check('3mm center + right beats left', identity(), thick3, [3,1,1], [1,1,1], false, [7,9,9], [9])
  const thick3Half = identity(); thick3Half[0][0] = 3; thick3Half[0][3] = 1.5
  check('3mm odd-dim half-integer center spans {0,1,2,3} symmetrically without +0.5 shift', identity(), thick3Half, [5,1,1], [1,1,1], false, [8,8,8,1,9], [8])
  check('3mm odd-dim half-integer center excludes slice 4 (no +0.5 shift)', identity(), thick3Half, [5,1,1], [1,1,1], false, [8,1,2,8,8], [8])

  // 5 mm grid (L = 5.0 -> n = 6 at +/-{1/12, 3/12, 5/12}):
  //   integer center C -> {C-2, C-1, C, C, C+1, C+2}; half-integer center C+0.5 -> {C-2..C+3} (zero +0.5 shift!)
  const thick5 = identity(); thick5[0][0] = 5; thick5[0][3] = 2
  check('5mm integer center spans all 5 slices symmetrically (outer slices win 3-2)', identity(), thick5, [5,1,1], [1,1,1], false, [8,1,2,8,8], [8])
  check('5mm integer center has double weight and breaks 2-2 tie', identity(), thick5, [5,1,1], [1,1,1], false, [4,4,9,6,6], [9])
  const thick5Half = identity(); thick5Half[0][0] = 5; thick5Half[0][3] = 2.5
  check('5mm odd-dim half-integer center spans {0,1,2,3,4,5} symmetrically without +0.5 shift', identity(), thick5Half, [7,1,1], [1,1,1], false, [8,8,8,8,1,2,9], [8])
  check('5mm odd-dim half-integer center includes slice 0 and excludes slice 6', identity(), thick5Half, [7,1,1], [1,1,1], false, [8,1,2,3,8,8,9], [8])

  // 7 mm grid (L = 7.0 -> n = 8 at +/-{1/16, 3/16, 5/16, 7/16})
  const thick7 = identity(); thick7[0][0] = 7; thick7[0][3] = 3
  check('7mm integer center spans all 7 slices symmetrically', identity(), thick7, [7,1,1], [1,1,1], false, [5,1,2,3,4,5,5], [5])
  const thick7Half = identity(); thick7Half[0][0] = 7; thick7Half[0][3] = 3.5
  check('7mm odd-dim half-integer center spans {0..7} symmetrically', identity(), thick7Half, [8,1,1], [1,1,1], false, [6,6,1,2,3,4,6,6], [6])

  check('identity', identity(), identity(), [9,7,5], [9,7,5])
  for (const t of [-12,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,12]) {
    const b=identity(); b[0][3]=t; b[1][3]=-t
    check(`translation and half-rounding ${t}`, identity(), b)
  }
  for (let n=0;n<100;n++) {
    const a=identity(), b=identity()
    for (let i=0;i<3;i++) {
      a[i][i]=(random()<0.5?-1:1)*(0.75+random()*2)
      b[i][i]=(random()<0.5?-1:1)*(0.75+random()*5.5)
      a[i][3]=Math.fround(random()*4-2); b[i][3]=Math.fround(random()*4-2)
      a[i][(i+1)%3]=Math.fround(random()*0.7); b[i][(i+2)%3]=Math.fround(random()*0.7)
    }
    check(`oblique/anisotropic/flip ${n}`, a.map(r=>r.map(Math.fround)), b.map(r=>r.map(Math.fround)))
  }
  // 60 dedicated thin/isotropic oblique 3D rotation cases (all axes L <= 1.5) to
  // hammer the streaming 2x2x2 fixed-point trilinear C <-> JS parity path.
  for (let n=0;n<60;n++) {
    const ux = random()*2-1, uy = random()*2-1, uz = random()*2-1
    const un = Math.hypot(ux, uy, uz) || 1
    const x = ux/un, y = uy/un, z = uz/un
    const th = (2 + random()*28) * Math.PI / 180
    const c = Math.cos(th), s = Math.sin(th), omc = 1 - c
    const R = [
      [c + x*x*omc,     x*y*omc - z*s, x*z*omc + y*s],
      [y*x*omc + z*s,   c + y*y*omc,   y*z*omc - x*s],
      [z*x*omc - y*s,   z*y*omc + x*s, c + z*z*omc],
    ]
    const sx = (random()<0.5?-1:1)*(0.6 + random()*0.75)
    const sy = (random()<0.5?-1:1)*(0.6 + random()*0.75)
    const sz = (random()<0.5?-1:1)*(0.6 + random()*0.75)
    const b = [
      [Math.fround(R[0][0]*sx), Math.fround(R[0][1]*sy), Math.fround(R[0][2]*sz), Math.fround(random()*3-1.5)],
      [Math.fround(R[1][0]*sx), Math.fround(R[1][1]*sy), Math.fround(R[1][2]*sz), Math.fround(random()*3-1.5)],
      [Math.fround(R[2][0]*sx), Math.fround(R[2][1]*sy), Math.fround(R[2][2]*sz), Math.fround(random()*3-1.5)],
      [0, 0, 0, 1],
    ]
    check(`oblique thin/iso trilinear ${n}`, identity(), b, [11,10,9], [9,8,7])
  }
  check('axis permutation', identity(), [[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
  const singular=identity(); singular[1]=[0,0,0,0]
  check('singular source', singular, identity(), undefined, undefined, true)
  check('singular destination', identity(), singular, undefined, undefined, true)
  const nonfinite=identity(); nonfinite[0][3]=NaN
  check('nonfinite affine', identity(), nonfinite, undefined, undefined, true)
  console.log(`native label resampling: ${cases} actual-browser oracle cases PASS`)
} finally { rmSync(dir, {recursive:true, force:true}) }
