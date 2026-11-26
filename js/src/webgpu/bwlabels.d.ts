export declare class BWLabeler {
  bwlabel(
    img: ArrayLike<number>,
    dim: number[],
    conn?: number,
    binarize?: boolean,
    onlyLargestClusterPerClass?: boolean,
    scratch?: Record<string, unknown> | null,
  ): [number, Uint32Array]
  filter18ClassFromLabels(
    bw: Uint8Array | Uint32Array,
    cl: number,
    ls: Uint32Array,
    dim: number[],
    targetClasses: Set<number>,
    relabelSuppressed?: boolean,
    familyOf?: ((cls: number) => number) | null,
    inPlace?: boolean,
  ): [number, Uint8Array | Uint32Array]
  filterHierarchicalFromLabels(
    bw: Uint8Array | Uint32Array,
    cl: number,
    ls: Uint32Array,
    dim: number[],
    targetClasses: 'all' | Set<number>,
    largerIndexWins?: boolean,
    relabelSuppressed?: boolean,
    familyOf?: ((cls: number) => number) | null,
    inPlace?: boolean,
  ): [number, Uint8Array | Uint32Array]
}

export declare function largestBinaryComponent26<T extends Uint8Array | Uint32Array | Float32Array>(
  data: T,
  dim: number[],
  binaryOutput?: boolean,
): T

export declare function growMaskBorderMm<T extends Uint8Array | Float32Array>(
  mask: T,
  dim: number[],
  dx: number,
  dy: number,
  dz: number,
  millimeters: number,
): T

export declare function releaseBwLabelScratch(): void
