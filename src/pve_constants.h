#ifndef BRAINCHOP_PVE_CONSTANTS_H
#define BRAINCHOP_PVE_CONSTANTS_H
/* Shared with the 18-class CAT-lite probability head (v1.1.1 / revision 2).
 * Label 0 (unknown) is background. Label 13 (Brain-Stem) joins WM_MASK so
 * brain_wm and brain_brain have no brainstem hole. Myelinated subcortical
 * structures {7: Thalamus, 10: Pallidum, 17: VentralDC} split 50/50 between
 * GM and WM priors (BC_PRIOR_MIXED_GWM_MASK) so brain_brain retains full
 * support while CAT-lite's purity gate keeps them from skewing cortical
 * mu_gm / sd_gm. */
#define BC_PVE_PRIOR_REVISION 2
#define BC_PRIOR_TEMPERATURE 2.0f
#define BC_PRIOR_SUPPORT_TEMPERATURE 1.0f
#define BC_PRIOR_GM_MASK ((1u<<2)|(1u<<6)|(1u<<8)|(1u<<9)| \
                        (1u<<14)|(1u<<15)|(1u<<16))
#define BC_PRIOR_WM_MASK ((1u<<1)|(1u<<5)|(1u<<13))
#define BC_PRIOR_MIXED_GWM_MASK ((1u<<7)|(1u<<10)|(1u<<17))
#define BC_PRIOR_CSF_MASK ((1u<<3)|(1u<<4)|(1u<<11)|(1u<<12))
#define BC_PRIOR_PLANES 4
#endif
