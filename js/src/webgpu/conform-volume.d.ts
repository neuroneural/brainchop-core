export declare const DISPLAY_CONFORM_FHIGH: number
export declare const INFERENCE_CONFORM_FHIGH: number
export declare function hounsfieldToCormackInPlace(vol: Float32Array): void
export declare function scaleIntensityFullRangeInPlace(vol: Float32Array): void
export declare function conformVolumeData(params: {
  img: ArrayBuffer | ArrayBufferView
  datatypeCode: number
  dims: number[]
  affine: number[] | number[][]
  sclSlope?: number
  sclInter?: number
  toRAS?: boolean
  isLinear?: boolean
  asFloat32?: boolean
  fHigh?: number
  inferenceFHigh?: number | null
  ct?: boolean
  comply?: boolean
  rejectConstant?: boolean
}): {
  img: Uint8Array | Float32Array
  inferenceImg?: Uint8Array | null
  dims: number[]
  pixDims: number[]
  affine: number[]
  datatypeCode: number
  bitsPerVoxel: number
  sclSlope: number
  sclInter: number
  calMin: number
  calMax: number
}
