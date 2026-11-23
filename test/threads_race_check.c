/*
 * A ThreadSanitizer gate for bc_parallel_for, with its own positive control.
 *
 * WHY THIS EXISTS. The pool's correctness is otherwise asserted entirely
 * through outputs -- `make test`, the thread-count sweep, the browser gates --
 * and a bit-exact test structurally CANNOT detect a benign race: two threads
 * writing the same value pass every time. That was true of the OpenMP loops
 * this replaced and that holds by design; it is true of the pool as well.
 *
 * WHY IT HAS TWO MODES, which is the whole point. A TSan run that reports
 * nothing is worthless unless you know the detector was armed and could see
 * through the pool at all. So this asserts BOTH directions:
 *
 *   clean  every write is index-disjoint, the shape every real call site uses.
 *          TSan must report NOTHING. A failure here is a real defect.
 *   race   one deliberately unsynchronised counter, everything else identical.
 *          TSan MUST report it. A pass here means the gate is not armed --
 *          wrong build flags, an un-instrumented pool, or a pool that quietly
 *          ran everything on one thread -- and the clean run above proved
 *          nothing.
 *
 * The second mode is what stops this being a test that passes while testing
 * nothing. It caught nothing when written; it exists so that a future change to
 * bc_threads.c's build cannot silently disarm the first.
 *
 * Run it through `make tsan-threads`, which builds and runs both modes in ONE
 * make invocation. Two invocations would rebuild uninstrumented under the
 * second command's flags and report green.
 */
#include "bc_threads.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SPAN 256
#define GATE_SECONDS 10

/*
 * THE RENDEZVOUS, and it is the reason this harness is trustworthy at all.
 *
 * Without it both modes are scheduler-dependent, and measurably so -- benchmarking
 * has the rate. bc_parallel_for makes the calling thread a participant, and 256
 * indices at grain 4 is only 64 chunks, so a caller that gets going before any
 * worker wakes can drain the entire range itself.
 *
 * That breaks BOTH modes, in opposite and equally silent ways:
 *   race   the planted counter is touched by one thread, so there is no race,
 *          TSan reports nothing, and the target FAILS A CORRECT POOL -- a ~15%
 *          flake, and worse on a loaded CI runner.
 *   clean  nothing concurrent ran, so "TSan found no race" is vacuous and the
 *          target PASSES WITHOUT TESTING ANYTHING.
 *
 * So the first thread into the callback blocks until a second arrives. That is
 * self-correcting rather than a sleep: a blocked caller is not draining, so the
 * remaining chunks stay available for a worker to claim. The timeout is a
 * deadlock guard, not a race window -- if it fires, no worker ever entered,
 * which is a real failure and is reported as one.
 *
 * It deliberately does NOT hide the planted race. The mutex orders the
 * rendezvous sections against each other, but every shared_counter increment
 * happens after the lock is released, so no happens-before edge covers them.
 * Verified: TSan still reports the race in `race` mode.
 */
static pthread_mutex_t gate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cond = PTHREAD_COND_INITIALIZER;
static int gate_open;
static int gate_timeout;   /* a waiter gave up; see the assertion in main */
static int gate_distinct;

/* Record this thread, then wait for a second one. */
static void rendezvous(void) {
	pthread_mutex_lock(&gate_lock);
	/* One thread-local flag beats scanning an array of pthread_t. */
	static _Thread_local int counted;
	if (!counted) { counted = 1; gate_distinct++; }
	/*
	 * !gate_timeout matters as much as !gate_open: without it the guard is
	 * PER CHUNK, not per run. A thread that times out leaves gate_open at 0,
	 * so its next chunk waits the full period again -- 64 chunks at grain 4,
	 * so a 10 s guard becomes ~170 s per mode and can outlive the CI job's
	 * 15-minute limit, turning the failure this exists to DIAGNOSE into a
	 * killed job with no message. Measured at 17 s against an intended 1 s.
	 */
	if (!gate_open && !gate_timeout) {
		if (gate_distinct >= 2) {
			gate_open = 1;
			pthread_cond_broadcast(&gate_cond);
		} else {
			struct timespec deadline;
			clock_gettime(CLOCK_REALTIME, &deadline);
			deadline.tv_sec += GATE_SECONDS;
			while (!gate_open)
				if (pthread_cond_timedwait(&gate_cond, &gate_lock, &deadline)) {
					/* Recorded, not swallowed. Without this the final check
					   tests ARRIVAL rather than OVERLAP: a waiter that gives up
					   and finishes its chunks, followed much later by a second
					   thread, still leaves gate_distinct == 2. */
					gate_timeout = 1;
					break;
				}
		}
	}
	pthread_mutex_unlock(&gate_lock);
}

static int shared_counter;        /* only touched in "race" mode */
static int visits[SPAN];
static int racy_mode;

/*
 * visits[] counts EXECUTIONS per index rather than storing a value, and that is
 * a deliberate correction: an earlier version wrote `disjoint[i] = i * 2` and
 * checked the final value, which proved nothing about coverage. A slot written
 * twice holds the same value as a slot written once, so duplicate execution
 * passed -- and index 0's expected value of 0 equals its static initializer, so
 * SKIPPING it passed too. Counting catches duplication, omission and the
 * zero-initializer collision with one assertion.
 *
 * These increments are themselves unsynchronized, which is correct here: if the
 * pool ever hands the same index to two threads, that IS a race and TSan should
 * say so in `clean` mode. The counter check and TSan are testing the same
 * property from two directions.
 */
static void body(void *context, int low, int high) {
	(void)context;
	rendezvous();
	for (int i = low; i <= high; i++) {
		/* The pattern every real call site uses: this thread owns index i and
		   no other thread will touch it. */
		visits[i]++;
		/* The planted defect, present only in the control run. */
		if (racy_mode) shared_counter++;
	}
}

int main(int argc, char **argv) {
	racy_mode = argc > 1 && !strcmp(argv[1], "race");

	int participants = bc_threads_count();
	printf("threads_race_check: mode=%s participants=%d\n",
		racy_mode ? "race" : "clean", participants);
	if (participants < 2) {
		/* Not a pass. With one participant there is no concurrency, so neither
		   mode proves anything -- including the control, whose whole job is to
		   fail. */
		printf("threads_race_check: FAIL -- pool gave %d participant(s); "
			"nothing concurrent ran\n", participants);
		return 1;
	}

	/* Grain 4, matching MN_GRAIN, so the claim path under test is the one the
	   convolution actually uses. */
	bc_parallel_for(0, SPAN - 1, 4, body, NULL);

	/*
	 * Overlap is asserted, not hoped for. Without this both modes can pass or
	 * fail on scheduling alone -- see the rendezvous comment above.
	 */
	printf("threads_race_check: %d distinct thread(s) entered the callback\n",
		gate_distinct);
	if (gate_timeout) {
		printf("threads_race_check: FAIL -- a thread waited %d s inside the "
			"callback without a second one arriving, so no overlap was proven "
			"even if others arrived later\n", GATE_SECONDS);
		return 1;
	}
	if (gate_distinct < 2) {
		printf("threads_race_check: FAIL -- only %d thread entered the callback "
			"in %d seconds, so nothing concurrent ran and neither mode proves "
			"anything\n", gate_distinct, GATE_SECONDS);
		return 1;
	}

	for (int i = 0; i < SPAN; i++) {
		if (visits[i] != 1) {
			printf("threads_race_check: FAIL -- index %d ran %d times, "
				"expected exactly 1; the range was %s\n", i, visits[i],
				visits[i] == 0 ? "not fully covered" : "covered more than once");
			return 1;
		}
	}

	/* Reported, not asserted. Under a real race this is < SPAN on a loaded
	   machine and can legitimately be SPAN on a lucky one, so it is evidence
	   for a human rather than a gate -- TSan is the gate. */
	if (racy_mode)
		printf("threads_race_check: counter=%d of %d (lost updates are the "
			"race being visible)\n", shared_counter, SPAN);
	printf("threads_race_check: range covered exactly once\n");
	return 0;
}
