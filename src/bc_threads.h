/*
 * A parallel-for over a one-dimensional inclusive range, replacing OpenMP.
 *
 * WHY THIS EXISTS. Every `#pragma omp` this project ever had was the same
 * construct -- `parallel for` over a single loop, with no reduction, no
 * critical section, no atomic, no barrier, no collapse, no simd, no tasks and
 * no nested parallelism. Ten regions, one shape.
 *
 * It cost a large fraction of each executable and, more to the point, made
 * Homebrew's libomp a hard build dependency on macOS, since Apple ships no
 * OpenMP runtime. pthreads is in libSystem. the project's *Threading is pthreads
 * now* owns the measurements; none is restated here, because a byte count
 * beside code that changes is a byte count that goes stale -- this comment
 * carried three superseded figures before an audit caught them, and a fourth
 * rounding survived the cleanup that was supposed to remove them.
 *
 * WHAT CALLERS MUST GUARANTEE. fn(context, low, high) is invoked with disjoint
 * sub-ranges of [low, high] that together cover it exactly, on an unspecified
 * number of threads in an unspecified order. So every call site must be a pure
 * map or gather whose output for one index depends on no other index. That was
 * already true of all ten -- it is the property the OpenMP version depended
 * on too, and the reason the CPU path is bit-exact regardless of thread count.
 * mn_moments looks like the exception and is not: it writes per-z partials to
 * disjoint slots and folds them serially afterwards, outside the parallel call.
 *
 * THREE RESTRICTIONS, all of the same kind: this is an internal facility for a
 * single-threaded program, not a reusable parallel-for library. Each is cheap
 * to honour and expensive to add machinery for, so they are stated rather than
 * enforced.
 *
 * 1. NO NESTING. bc_parallel_for must not be called from inside fn. The pool
 *    would deadlock waiting for itself.
 *
 * 2. ONE CALLER. Only one thread may be inside bc_parallel_for at a time, and
 *    bc_threads_set_count must be called before any of them. The pool has one
 *    set of job state -- fn, context, bounds, generation, running -- and the
 *    mutex protects individual updates to it, NOT a region as a whole. A second
 *    application thread entering while a region is live would overwrite that
 *    state, mixing callbacks or hanging the completion wait. Lazy init and
 *    bc_threads_set_count are unsynchronized for the same reason.
 *
 *    Every caller today is the main thread, and no path introduces a second:
 *    the CLI is single-threaded, and the wasm modules and the Windows CUDA DLL
 *    compile the serial fallback below. Note why serializing whole
 *    regions was rejected rather than merely skipped.
 *
 * 3. NO FORK once the pool exists. A forked child inherits pool.spawned counting
 *    workers that did not survive the fork, so its next bc_parallel_for waits
 *    forever for completions that cannot arrive. Nothing here forks, and
 *    pthread_atfork handlers are deliberately not added for a CLI that never
 *    will. Revisit only if this is ever embedded in a process that does.
 */
#ifndef BC_THREADS_H
#define BC_THREADS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Process the inclusive sub-range [low, high]. The whole int range is valid,
 * `high == INT_MAX` included: the pool keeps its cursor, bound and grain in
 * int64_t precisely so that advancing one past the last index cannot overflow.
 * An earlier revision did this arithmetic in int and required `high < INT_MAX`;
 * that contract is gone with the defect, and this comment used to state it
 * after the code had stopped needing it.
 *
 * The widest caller is ACCEL_NVOX = 256^3 = 16777216, which backend_accel.c
 * still asserts at compile time -- that assert is now about its own cast to
 * int, not about protecting this API.
 *
 * One caveat the pool cannot enforce: a callback written as the usual
 * `for (int i = low; i <= high; i++)` is itself UB at high == INT_MAX, since
 * the increment overflows on the last iteration. Every callback here is one of
 * those, and every one is bounded far below it. A caller that genuinely wants
 * the top of the range must iterate in a wider type.
 */
typedef void (*bc_parallel_fn)(void *context, int low, int high);

/*
 * Run fn over [low, high] inclusive, across bc_threads_count() participants
 * including the calling thread.
 *
 * grain > 0 selects dynamic scheduling in chunks of that many iterations --
 * the equivalent of schedule(dynamic, grain), and what the convolution wants:
 * the z slices are equal-cost but a hybrid CPU mixes fast and slow cores, so a
 * static split leaves the efficiency cores as stragglers.
 *
 * grain <= 0 selects an even split, the equivalent of schedule(static), for
 * loops whose iterations are cheap and uniform enough that handing out work on
 * demand would cost more than the imbalance it removes.
 *
 * An empty range (high < low) is a no-op. A single participant, or a build
 * without threading, calls fn(context, low, high) directly on this thread --
 * so the serial path costs one indirect call, not a pool round trip.
 */
void bc_parallel_for(int low, int high, int grain,
	bc_parallel_fn fn, void *context);

/*
 * Set the number of participants, including the calling thread. Values below 1
 * are ignored. This is what -cpu N drives; it must be called before any
 * bc_parallel_for, and the pool is sized on first use.
 *
 * Unset, the pool takes BC_NUM_THREADS when that is set to a whole positive
 * integer (anything else is ignored), else the CPUs this PROCESS may use --
 * sched_getaffinity on Linux, the machine's online count elsewhere.
 */
void bc_threads_set_count(long count);

/* The number of participants, resolving the default on first call. */
int bc_threads_count(void);

#ifdef __cplusplus
}
#endif

#endif
