/*
 * sched_getaffinity() and CPU_COUNT() are GNU extensions that -std=c11 hides,
 * so this preamble must precede every include. Linux only; nothing else here
 * needs it. Same shape as backend_accel.c's _POSIX_C_SOURCE.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

/*
 * The parallel-for pool. See bc_threads.h for the contract and for why this
 * replaced OpenMP.
 *
 * SHAPE OF THE DESIGN. A persistent pool of count-1 workers plus the calling
 * thread, woken once per parallel region and rejoined at its end. Work is
 * handed out as chunks of a shared counter, so `dynamic` and `static` are the
 * same code path differing only in chunk size -- static is "one chunk each".
 *
 * WHY A MUTEX AND NOT AN ATOMIC COUNTER. Claiming a chunk takes the pool mutex,
 * which looks like the obvious thing to make lock-free and is not worth it: the
 * lock is held for a dozen instructions and a whole run claims a few thousand
 * chunks, against regions two orders of magnitude longer (profiling has the
 * timings). Avoiding <stdatomic.h> is a second, forward-looking reason: it
 * keeps this compilable by MSVC for the ABI build named under *What this leaves
 * for Windows*. NOT for the CUDA DLL, which compiles the serial branch and
 * never sees this code at all.
 *
 * SINGLE CALLER, NO NESTING, NO FORK. bc_threads.h states these as the
 * facility's contract and gives the reasoning; the consequence HERE is that
 * neither the job state below nor start_pool() is guarded against a second
 * application thread, which is a deliberate choice and not an oversight.
 *
 * WHY NO SPIN-THEN-SLEEP. libomp keeps workers spinning between regions so a
 * wake costs nothing, which matters when regions are microseconds apart. These
 * are milliseconds apart, so a condition variable's wake latency is far below
 * the noise floor. Measure before adding one for a finer-grained call site.
 */
#include "bc_threads.h"

#ifndef BC_THREADS

/*
 * Threading compiled out -- the wasm builds, and OMP=0's successor THREADS=0
 * which the sanitizer targets use. Callers are unchanged; the range simply runs
 * whole on the calling thread.
 */
void bc_parallel_for(int low, int high, int grain,
		bc_parallel_fn fn, void *context) {
	(void)grain;
	if (high >= low) fn(context, low, high);
}

void bc_threads_set_count(long count) { (void)count; }

int bc_threads_count(void) { return 1; }

#else

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
/*
 * mingw-w64 ships <unistd.h>, so the include above compiles, but it declares no
 * sysconf and defines no _SC_NPROCESSORS_ONLN -- that pair is POSIX, and Windows
 * is not. The failure is a compile error rather than anything subtle, but it is
 * the ONE thing that stopped this file building there, so it is worth naming:
 * under OpenMP the count came from omp_get_max_threads() and no POSIX call was
 * ever reached. See resolve_count() for what replaces it.
 */
#ifdef _WIN32
#include <windows.h>
#endif

/*
 * An upper bound on participants, so a hostile BC_NUM_THREADS cannot ask for a
 * million stacks. Well above any machine this runs on; -cpu is clamped to it
 * silently because a thread count is a performance hint, not a correctness
 * input, and failing a segmentation over one would be the wrong trade.
 */
#define BC_THREADS_MAX 1024

/*
 * Statically initialized: PTHREAD_*_INITIALIZER is POSIX-guaranteed for
 * default-attribute objects, which is why start_pool() has one failure path and
 * no unwind ladder, and why nothing is destroyed at exit.
 *
 * mingw-w64's winpthreads provides both macros, and that leg has now been
 * compiled and run: MSYS2/UCRT64, gcc 16.1 and clang 22.1.8, both models
 * bit-exact against the macOS-generated fixtures. The one thing winpthreads did
 * NOT supply was sysconf(); see resolve_count().
 */
static struct {
	pthread_mutex_t mutex;
	pthread_cond_t ready;          /* a region has started, or shutdown */
	pthread_cond_t done;           /* a worker left the current region */
	pthread_t *worker;
	int spawned;                   /* threads created; participants = +1 */
	int count;                     /* participants, including the caller */
	int shutdown;
	unsigned long generation;      /* bumped once per region */
	int running;                   /* workers still inside this region */
	bc_parallel_fn fn;
	void *context;
	/*
	 * int64_t, though the API's bounds are int. The chunk arithmetic below adds
	 * grain to next and steps one past the last index, both of which are signed
	 * overflow at INT_MAX -- and overflow is UB, not wraparound, so a guard
	 * written as `if (last < next)` is one the compiler may delete. An audit
	 * demonstrated both with -fsanitize=undefined. Widening the state removes
	 * the question instead of guarding it, and costs nothing: these are three
	 * fields touched a few thousand times per run.
	 */
	int64_t next;                  /* lowest unclaimed iteration */
	int64_t high;
	int64_t grain;
} pool = {
	.mutex = PTHREAD_MUTEX_INITIALIZER,
	.ready = PTHREAD_COND_INITIALIZER,
	.done = PTHREAD_COND_INITIALIZER,
};

/* 0 uninitialised, 1 pool usable, -1 run everything on the calling thread. */
static int pool_state;
static long pool_requested;

/*
 * Claim the next chunk. Returns 0 once the region is exhausted, which is what
 * ends both the workers' loop and the calling thread's.
 */
static int claim(int *low, int *high) {
	int taken = 0;
	pthread_mutex_lock(&pool.mutex);
	if (pool.next <= pool.high) {
		int64_t last = pool.next + pool.grain - 1;
		if (last > pool.high) last = pool.high;
		*low = (int)pool.next;
		*high = (int)last;
		pool.next = last + 1;
		taken = 1;
	}
	pthread_mutex_unlock(&pool.mutex);
	return taken;
}

static void drain(bc_parallel_fn fn, void *context) {
	int low, high;
	while (claim(&low, &high)) fn(context, low, high);
}

static void *worker_main(void *argument) {
	unsigned long seen = 0;
	(void)argument;
	pthread_mutex_lock(&pool.mutex);
	for (;;) {
		while (!pool.shutdown && pool.generation == seen)
			pthread_cond_wait(&pool.ready, &pool.mutex);
		if (pool.shutdown) break;
		seen = pool.generation;
		/*
		 * Copy the region out under the lock. A worker cannot miss a
		 * generation: bc_parallel_for does not return until running hits 0, so
		 * no second region can be published while this one is outstanding.
		 */
		bc_parallel_fn fn = pool.fn;
		void *context = pool.context;
		pthread_mutex_unlock(&pool.mutex);
		drain(fn, context);
		pthread_mutex_lock(&pool.mutex);
		if (--pool.running == 0) pthread_cond_signal(&pool.done);
	}
	pthread_mutex_unlock(&pool.mutex);
	return NULL;
}

static void stop_pool(void) {
	if (pool_state != 1) return;
	pthread_mutex_lock(&pool.mutex);
	pool.shutdown = 1;
	pthread_cond_broadcast(&pool.ready);
	pthread_mutex_unlock(&pool.mutex);
	for (int i = 0; i < pool.spawned; i++) pthread_join(pool.worker[i], NULL);
	free(pool.worker);
	/* No state reset after this: stop_pool runs once, from atexit, and nothing
	   can observe the pool afterwards. Restore it if a second caller appears. */
	pool_state = -1;
}

/*
 * ONE OWNER for the clamp. Every path produces a raw long and this is the only
 * place a bound is applied; bc_threads_set_count and brainchopc.c deliberately
 * do not. (benchmarking records what having four owners cost.)
 */
static int resolve_count(void) {
	long wanted = pool_requested;
	if (wanted <= 0) {
		const char *text = getenv("BC_NUM_THREADS");
		char *end;
		if (text && *text) {
			long value = strtol(text, &end, 10);
			if (end != text && *end == '\0' && value > 0) wanted = value;
		}
	}
	/*
	 * How many CPUs this PROCESS may use, which is not the machine's core count.
	 *
	 * sysconf(_SC_NPROCESSORS_ONLN) reports the machine and ignores both the
	 * affinity mask and any cgroup quota, so `taskset -c 0,1` or a
	 * container-limited CI runner got a pool sized for the whole host with its
	 * workers fighting over a slice of it. sched_getaffinity answers the right
	 * question on Linux, which is the platform where both of those are common.
	 *
	 * It degrades rather than guesses: any failure falls through to the sysconf
	 * value, which is what every platform without it uses anyway. That matters
	 * because this was written on a Mac, where the __linux__ branch is not even
	 * compiled -- the failure mode of getting it wrong is the behaviour that was
	 * already shipping, not a broken build.
	 *
	 * NOT closed by this, both deliberately:
	 *
	 * - cgroup CPU QUOTA (cpu.max) limits CPU-seconds rather than which CPUs,
	 *   so it does not appear in the affinity mask at all and a quota-limited
	 *   container still oversubscribes. Reading it means parsing cgroup v1 and
	 *   v2 paths.
	 * - A mask wider than CPU_SETSIZE (1024) makes sched_getaffinity return
	 *   EINVAL, and this falls back to the host-wide count -- so a process
	 *   pinned to a few cores of a >1024-CPU machine is sized as badly as it
	 *   was before this branch existed. CPU_ALLOC and a retry loop would fix
	 *   it. ponytail: not written, for hardware nobody here can test on; the
	 *   failure is the old behaviour, not a new one.
	 *
	 * Both are in Known gaps, and -cpu overrides either.
	 */
#ifdef __linux__
	if (wanted <= 0) {
		cpu_set_t affinity;
		if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
			int usable = CPU_COUNT(&affinity);
			if (usable > 0) wanted = usable;
		}
	}
#endif
	/*
	 * The Windows answer to the same question the __linux__ branch above asks.
	 * GetProcessAffinityMask is the sched_getaffinity analogue -- it reports the
	 * CPUs this PROCESS may use, so `start /affinity 3` is honoured -- and
	 * GetSystemInfo is the sysconf(_SC_NPROCESSORS_ONLN) analogue underneath it.
	 *
	 * Both are per PROCESSOR GROUP, and that is a limitation shared with the
	 * default this replaces rather than a new one: a thread created without
	 * GROUP_AFFINITY stays in its creator's group, so a pool sized by the group's
	 * width is sized by what its threads can actually run on. Above 64 CPUs the
	 * machine has more, and this asks for the ceiling that applies. The affinity
	 * call is the one that can fail (it does for a multi-group process), which is
	 * why the count falls through to GetSystemInfo rather than to 1.
	 */
#ifdef _WIN32
	if (wanted <= 0) {
		DWORD_PTR process_mask, system_mask;
		if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
			int usable = 0;
			for (DWORD_PTR bit = process_mask; bit; bit &= bit - 1) usable++;
			if (usable > 0) wanted = usable;
		}
	}
	if (wanted <= 0) {
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		wanted = (long)info.dwNumberOfProcessors;
	}
#else
	if (wanted <= 0) wanted = sysconf(_SC_NPROCESSORS_ONLN);
#endif
	if (wanted < 1) return 1;
	return wanted > BC_THREADS_MAX ? BC_THREADS_MAX : (int)wanted;
}

/*
 * Size the pool on first use. Every failure here degrades to running on the
 * calling thread rather than failing the segmentation -- a thread count is a
 * performance hint. A partial pthread_create failure keeps the threads that did
 * start, so the run is slower rather than dead.
 */
static int start_pool(void) {
	if (pool_state) return pool_state == 1;
	pool_state = -1;
	pool.count = 1;

	int wanted = resolve_count();
	if (wanted <= 1) return 0;
	pool.worker = (pthread_t *)malloc((size_t)(wanted - 1) * sizeof(*pool.worker));
	if (!pool.worker) return 0;
	for (int i = 0; i < wanted - 1; i++) {
		if (pthread_create(&pool.worker[i], NULL, worker_main, NULL)) break;
		pool.spawned++;
	}
	if (pool.spawned == 0) {
		free(pool.worker);
		pool.worker = NULL;
		return 0;
	}
	pool.count = pool.spawned + 1;
	pool_state = 1;
	/*
	 * Join at exit rather than leaking the threads and their stacks. Not
	 * cosmetic: the sanitizer targets run LeakSanitizer on Linux, and this is
	 * the only allocation in the program that outlives main by design.
	 */
	/* If registration fails the workers simply outlive main, which is what
	   OpenMP did anyway; it is not worth failing a segmentation over. */
	(void)atexit(stop_pool);
	return 1;
}

void bc_parallel_for(int low, int high, int grain,
		bc_parallel_fn fn, void *context) {
	if (high < low) return;
	if (!start_pool()) {
		fn(context, low, high);
		return;
	}

	/*
	 * grain <= 0 is schedule(static): one chunk per participant. Rounding up
	 * can leave the last participant idle, which is what static scheduling
	 * does anyway and is why these call sites chose it.
	 */
	/* int64_t throughout: `span + count - 1` overflows int for a range near
	   INT_MAX, which an audit demonstrated under UBSan. See the field comment. */
	int64_t span = (int64_t)high - low + 1;
	int64_t chunk = grain;
	if (chunk <= 0) chunk = (span + pool.count - 1) / pool.count;
	if (chunk < 1) chunk = 1;

	pthread_mutex_lock(&pool.mutex);
	pool.fn = fn;
	pool.context = context;
	pool.next = low;
	pool.high = high;
	pool.grain = chunk;
	pool.running = pool.spawned;
	pool.generation++;
	pthread_cond_broadcast(&pool.ready);
	pthread_mutex_unlock(&pool.mutex);

	/* The calling thread is a participant, not an idle supervisor. */
	drain(fn, context);

	pthread_mutex_lock(&pool.mutex);
	while (pool.running > 0) pthread_cond_wait(&pool.done, &pool.mutex);
	pthread_mutex_unlock(&pool.mutex);
}

void bc_threads_set_count(long count) {
	/*
	 * Stored raw; resolve_count() owns the clamp. A call after the pool has been
	 * sized cannot take effect, and would otherwise be a silent no-op no test
	 * could catch -- the CPU path is thread-count independent by design -- so it
	 * says so instead.
	 */
	if (pool_state != 0) {
		fprintf(stderr, "brainchopc: warning: thread count set after the pool "
			"was sized; keeping %d\n", pool.count);
		return;
	}
	if (count > 0) pool_requested = count;
}

int bc_threads_count(void) {
	if (!start_pool()) return 1;
	return pool.count;
}

#endif
