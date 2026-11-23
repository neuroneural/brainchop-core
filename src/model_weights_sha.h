/* First 8 lowercase hex characters of SHA-256(src/models/<model>/model.bin).
 * Checked against each embedded model.bin by `make -C src check-payloads`
 * and `make -C src test-models` so NIfTI descrip provenance cannot drift
 * from the shipped weights. */
#ifndef BRAINCHOPC_MODEL_WEIGHTS_SHA_H
#define BRAINCHOPC_MODEL_WEIGHTS_SHA_H

#if defined(BC_MODEL_IS_MINDGRAB)
#define BC_MODEL_WEIGHTS_SHA8 "c9a01490"
#elif defined(BC_MODEL_IS_MINDMAP)
#define BC_MODEL_WEIGHTS_SHA8 "58006d19"
#elif defined(BC_MODEL_IS_MINDSNAP)
#define BC_MODEL_WEIGHTS_SHA8 "995cdb52"
#elif defined(BC_MODEL_IS_MODEL16CHAN18CLS)
#define BC_MODEL_WEIGHTS_SHA8 "22d73f43"
#else
#define BC_MODEL_WEIGHTS_SHA8 "unknown"
#endif

#endif
