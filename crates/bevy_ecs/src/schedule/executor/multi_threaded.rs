use alloc::{boxed::Box, vec::Vec};
use arrayvec::ArrayVec;
use bevy_platform::cell::SyncUnsafeCell;
use bevy_platform::sync::Arc;
use bevy_tasks::{ComputeTaskPool, Scope, TaskPool, ThreadExecutor};
use concurrent_queue::ConcurrentQueue;
use core::{
    any::Any,
    panic::AssertUnwindSafe,
    sync::atomic::{fence, AtomicUsize, Ordering},
};
use fixedbitset::FixedBitSet;
#[cfg(feature = "std")]
use std::eprintln;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

/// How long an idle worker keeps polling the claimable queue while other systems are still
/// running, before it retires. Completions usually release the next systems within a few
/// microseconds, and a retired worker costs a task spawn to replace.
const WORKER_LINGER: core::time::Duration = core::time::Duration::from_micros(0);

/// How many workers a dispatch pass spawns directly when too few are alive; the workers
/// themselves add more while work remains queued.
const DISPATCH_SEED_WORKERS: usize = usize::MAX;

/// `in_flight` only serves the linger heuristic; with lingering disabled it is not maintained,
/// which keeps a contended read-modify-write off every dispatch and completion.
const TRACK_IN_FLIGHT: bool = !WORKER_LINGER.is_zero();

/// Whether workers that find more work queued behind their claim spawn one more worker.
const TREE_SPAWN: bool = false;

/// Whether a worker that finds the executor lock free processes its own completion in place
/// (instead of always handing it over through the completion queue).
const FAST_PATH_COMPLETION: bool = true;

/// The most systems one worker claims with a single queue operation.
///
/// Kept at one: ready, non-conflicting systems must be able to run concurrently while
/// threads are free (Bevy's `parallel_execution` test synchronises such systems through a
/// `Barrier`), and a worker that has claimed several runs them in order, so a claim of
/// more than one could hold back a system that another one is waiting on.
const MAX_CLAIM: usize = 1;

/// Puts a hot atomic on its own cache line so spinning readers and writers of different
/// counters do not fight over one line.
#[repr(align(128))]
struct Padded<T>(T);

/// The queue of systems waiting for a worker.
///
/// One slot per system of the schedule; entries are pushed only by the thread holding the
/// executor lock (so there is a single producer at any time) and claimed by workers in
/// ranges, so a wide phase costs one compare-and-swap per batch instead of one per system.
struct ClaimQueue {
    slots: Box<[AtomicUsize]>,
    head: Padded<AtomicUsize>,
    tail: Padded<AtomicUsize>,
}

impl ClaimQueue {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| AtomicUsize::new(usize::MAX)).collect(),
            head: Padded(AtomicUsize::new(0)),
            tail: Padded(AtomicUsize::new(0)),
        }
    }

    /// Rewinds the queue for a new run. Only sound while no worker is alive.
    fn reset(&self) {
        self.head.0.store(0, Ordering::Relaxed);
        self.tail.0.store(0, Ordering::Relaxed);
    }

    /// Appends a system. Only called while holding the executor lock.
    fn push(&self, system_index: usize) {
        let tail = self.tail.0.load(Ordering::Relaxed);
        self.slots[tail].store(system_index, Ordering::Relaxed);
        self.tail.0.store(tail + 1, Ordering::Release);
    }

    fn len(&self) -> usize {
        let tail = self.tail.0.load(Ordering::Acquire);
        let head = self.head.0.load(Ordering::Acquire);
        tail.saturating_sub(head)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Claims up to `max` queued systems, returning the range of slots now owned by the caller
    /// and how many entries were still queued behind it at that moment.
    fn claim(&self, max: usize) -> Option<(core::ops::Range<usize>, usize)> {
        let mut head = self.head.0.load(Ordering::Acquire);
        let mut attempts: u32 = 0;
        loop {
            let tail = self.tail.0.load(Ordering::Acquire);
            if head >= tail {
                return None;
            }
            let count = max.min(tail - head);
            match self.head.0.compare_exchange_weak(
                head,
                head + count,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some((head..head + count, tail - head - count)),
                Err(actual) => {
                    head = actual;
                    // Back off exponentially on contention so that a crowd of claimers does
                    // not thrash the head line with immediate retries.
                    for _ in 0..(1u32 << attempts.min(6)) {
                        core::hint::spin_loop();
                    }
                    attempts += 1;
                }
            }
        }
    }

    /// Reads a claimed slot.
    fn get(&self, slot: usize) -> usize {
        self.slots[slot].load(Ordering::Relaxed)
    }
}

#[cfg(feature = "trace")]
use tracing::{info_span, Span};

use crate::{
    error::{ErrorContext, ErrorHandler, Result},
    prelude::Resource,
    schedule::{
        is_apply_deferred, ConditionWithAccess, SystemExecutor, SystemSchedule, SystemWithAccess,
    },
    system::{RunSystemError, ScheduleSystem},
    world::{unsafe_world_cell::UnsafeWorldCell, World},
};
#[cfg(feature = "hotpatching")]
use crate::{prelude::DetectChanges, HotPatchChanges};

use super::__rust_begin_short_backtrace;

/// Borrowed data used by the [`MultiThreadedExecutor`].
struct Environment<'env, 'sys> {
    executor: &'env MultiThreadedExecutor,
    systems: &'sys [SyncUnsafeCell<SystemWithAccess>],
    conditions: SyncUnsafeCell<Conditions<'sys>>,
    world_cell: UnsafeWorldCell<'env>,
}

struct Conditions<'a> {
    system_conditions: &'a mut [Vec<ConditionWithAccess>],
    set_conditions: &'a mut [Vec<ConditionWithAccess>],
    sets_with_conditions_of_systems: &'a [FixedBitSet],
    systems_in_sets_with_conditions: &'a [FixedBitSet],
}

impl<'env, 'sys> Environment<'env, 'sys> {
    fn new(
        executor: &'env MultiThreadedExecutor,
        schedule: &'sys mut SystemSchedule,
        world: &'env mut World,
    ) -> Self {
        Environment {
            executor,
            systems: SyncUnsafeCell::from_mut(schedule.systems.as_mut_slice()).as_slice_of_cells(),
            conditions: SyncUnsafeCell::new(Conditions {
                system_conditions: &mut schedule.system_conditions,
                set_conditions: &mut schedule.set_conditions,
                sets_with_conditions_of_systems: &schedule.sets_with_conditions_of_systems,
                systems_in_sets_with_conditions: &schedule.systems_in_sets_with_conditions,
            }),
            world_cell: world.as_unsafe_world_cell(),
        }
    }
}

/// Per-system data used by the [`MultiThreadedExecutor`].
// Copied here because it can't be read from the system when it's running.
struct SystemTaskMetadata {
    /// The set of systems whose `component_access_set()` conflicts with this one.
    conflicting_systems: FixedBitSet,
    /// The set of systems whose `component_access_set()` conflicts with this system's conditions.
    /// Note that this is separate from `conflicting_systems` to handle the case where
    /// a system is skipped by an earlier system set condition or system stepping,
    /// and needs access to run its conditions but not for itself.
    condition_conflicting_systems: FixedBitSet,
    /// Indices of the systems that directly depend on the system.
    dependents: Vec<usize>,
    /// Is `true` if the system does not access `!Send` data.
    is_send: bool,
    /// Is `true` if the system is exclusive.
    is_exclusive: bool,
}

/// The result of running a system that is sent across a channel.
struct SystemResult {
    system_index: usize,
}

/// Runs the schedule using a thread pool. Non-conflicting systems can run in parallel.
///
/// # Design
///
/// Scheduling decisions (dependency counting, access-conflict checks and run-condition
/// evaluation) happen under a mutex, exactly once per system. Systems that may run are
/// not spawned as individual tasks; instead they are pushed onto a lock-free *claimable*
/// queue and executed by a small number of long-lived *worker* tasks (at most one per
/// thread of the [`ComputeTaskPool`]). A worker pops the next claimable system, runs it,
/// reports its completion and immediately continues with the next one, so a chain of
/// dependent systems runs on one thread with no task allocation and no thread hand-off
/// between its links, while wide, conflict-free phases still fan out across the pool.
///
/// Workers never block on the scheduling mutex. Whichever thread holds it performs the
/// bookkeeping for every completion that arrived in the meantime; a worker that fails to
/// acquire it simply moves on to the next claimable system. Exclusive systems and systems
/// that need `!Send` data keep their own tasks on the scope / main thread as before.
pub struct MultiThreadedExecutor {
    /// The running state, protected by a mutex so that a reference to the executor can be shared across tasks.
    state: Mutex<ExecutorState>,
    /// Queue of system completion events.
    system_completion: ConcurrentQueue<SystemResult>,
    /// Systems that passed their conflict checks and run conditions and are waiting for a worker.
    claimable: ClaimQueue,
    /// Number of worker tasks currently alive.
    active_workers: Padded<AtomicUsize>,
    /// Systems dispatched but not yet completed (a lock-free mirror of
    /// `ExecutorState::num_running_systems`, so idle workers can tell whether more
    /// work may still arrive without taking the lock).
    in_flight: Padded<AtomicUsize>,
    /// Upper bound on the number of workers alive at once (the compute pool's thread count).
    max_workers: usize,
    /// Setting when true applies deferred system buffers after all systems have run
    apply_final_deferred: bool,
    /// When set, tells the executor that a thread has panicked.
    panic_payload: Mutex<Option<Box<dyn Any + Send>>>,
    starting_systems: FixedBitSet,
    /// Cached tracing span
    #[cfg(feature = "trace")]
    executor_span: Span,
}

/// The state of the executor while running.
pub struct ExecutorState {
    /// Metadata for scheduling and running system tasks.
    system_task_metadata: Vec<SystemTaskMetadata>,
    /// The set of systems whose `component_access_set()` conflicts with this system set's conditions.
    set_condition_conflicting_systems: Vec<FixedBitSet>,
    /// Returns `true` if a system with non-`Send` access is running.
    local_thread_running: bool,
    /// Returns `true` if an exclusive system is running.
    exclusive_running: bool,
    /// The number of systems that are running.
    num_running_systems: usize,
    /// The number of dependencies each system has that have not completed.
    num_dependencies_remaining: Vec<usize>,
    /// System sets whose conditions have been evaluated.
    evaluated_sets: FixedBitSet,
    /// Systems that have no remaining dependencies and are waiting to run.
    ready_systems: FixedBitSet,
    /// copy of `ready_systems`
    ready_systems_copy: FixedBitSet,
    /// Systems that are running.
    running_systems: FixedBitSet,
    /// Systems that got skipped.
    skipped_systems: FixedBitSet,
    /// Systems whose conditions have been evaluated and were run or skipped.
    completed_systems: FixedBitSet,
    /// Systems that have run but have not had their buffers applied.
    unapplied_systems: FixedBitSet,
    /// Spare bitsets handed to `ApplyDeferred` tasks so they do not allocate.
    unapplied_scratch: Vec<FixedBitSet>,
}

/// References to data required by the executor.
/// This is copied to each system task so that can invoke the executor when they complete.
// These all need to outlive 'scope in order to be sent to new tasks,
// and keeping them all in a struct means we can use lifetime elision.
#[derive(Copy, Clone)]
struct Context<'scope, 'env, 'sys> {
    environment: &'env Environment<'env, 'sys>,
    scope: &'scope Scope<'scope, 'env, ()>,
    error_handler: ErrorHandler,
}

impl Default for MultiThreadedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemExecutor for MultiThreadedExecutor {
    fn init(&mut self, schedule: &SystemSchedule) {
        let state = self.state.get_mut().unwrap();
        // pre-allocate space
        let sys_count = schedule.system_ids.len();
        let set_count = schedule.set_ids.len();

        self.system_completion = ConcurrentQueue::bounded(sys_count.max(1));
        self.claimable = ClaimQueue::with_capacity(sys_count);
        self.starting_systems = FixedBitSet::with_capacity(sys_count);
        state.evaluated_sets = FixedBitSet::with_capacity(set_count);
        state.ready_systems = FixedBitSet::with_capacity(sys_count);
        state.ready_systems_copy = FixedBitSet::with_capacity(sys_count);
        state.running_systems = FixedBitSet::with_capacity(sys_count);
        state.completed_systems = FixedBitSet::with_capacity(sys_count);
        state.skipped_systems = FixedBitSet::with_capacity(sys_count);
        state.unapplied_systems = FixedBitSet::with_capacity(sys_count);
        state.unapplied_scratch.clear();

        state.system_task_metadata = Vec::with_capacity(sys_count);
        for index in 0..sys_count {
            state.system_task_metadata.push(SystemTaskMetadata {
                conflicting_systems: FixedBitSet::with_capacity(sys_count),
                condition_conflicting_systems: FixedBitSet::with_capacity(sys_count),
                dependents: schedule.system_dependents[index].clone(),
                is_send: schedule.systems[index].system.is_send(),
                is_exclusive: schedule.systems[index].system.is_exclusive(),
            });
            if schedule.system_dependencies[index] == 0 {
                self.starting_systems.insert(index);
            }
        }

        {
            #[cfg(feature = "trace")]
            let _span = info_span!("calculate conflicting systems").entered();
            for index1 in 0..sys_count {
                let system1 = &schedule.systems[index1];
                for index2 in 0..index1 {
                    let system2 = &schedule.systems[index2];
                    if !system2.access.is_compatible(&system1.access) {
                        state.system_task_metadata[index1]
                            .conflicting_systems
                            .insert(index2);
                        state.system_task_metadata[index2]
                            .conflicting_systems
                            .insert(index1);
                    }
                }

                for index2 in 0..sys_count {
                    let system2 = &schedule.systems[index2];
                    if schedule.system_conditions[index1]
                        .iter()
                        .any(|condition| !system2.access.is_compatible(&condition.access))
                    {
                        state.system_task_metadata[index1]
                            .condition_conflicting_systems
                            .insert(index2);
                    }
                }
            }

            state.set_condition_conflicting_systems.clear();
            state.set_condition_conflicting_systems.reserve(set_count);
            for set_idx in 0..set_count {
                let mut conflicting_systems = FixedBitSet::with_capacity(sys_count);
                for sys_index in 0..sys_count {
                    let system = &schedule.systems[sys_index];
                    if schedule.set_conditions[set_idx]
                        .iter()
                        .any(|condition| !system.access.is_compatible(&condition.access))
                    {
                        conflicting_systems.insert(sys_index);
                    }
                }
                state
                    .set_condition_conflicting_systems
                    .push(conflicting_systems);
            }
        }

        state.num_dependencies_remaining = Vec::with_capacity(sys_count);
    }

    fn run(
        &mut self,
        schedule: &mut SystemSchedule,
        world: &mut World,
        _skip_systems: Option<&FixedBitSet>,
        error_handler: ErrorHandler,
    ) {
        let state = self.state.get_mut().unwrap();
        // reset counts
        if schedule.systems.is_empty() {
            return;
        }
        state.num_running_systems = 0;
        state
            .num_dependencies_remaining
            .clone_from(&schedule.system_dependencies);
        state.ready_systems.clone_from(&self.starting_systems);

        // If stepping is enabled, make sure we skip those systems that should
        // not be run.
        #[cfg(feature = "bevy_debug_stepping")]
        if let Some(skipped_systems) = _skip_systems {
            debug_assert_eq!(skipped_systems.len(), state.completed_systems.len());
            // mark skipped systems as completed
            state.completed_systems |= skipped_systems;

            // signal the dependencies for each of the skipped systems, as
            // though they had run
            for system_index in skipped_systems.ones() {
                state.signal_dependents(system_index);
                state.ready_systems.remove(system_index);
            }
        }

        let thread_executor = world
            .get_resource::<MainThreadExecutor>()
            .map(|e| e.0.clone());
        let thread_executor = thread_executor.as_deref();

        let pool = ComputeTaskPool::get_or_init(TaskPool::default);
        self.max_workers = pool.thread_num().max(1);
        self.in_flight.0.store(0, Ordering::Relaxed);
        debug_assert_eq!(self.active_workers.0.load(Ordering::Relaxed), 0);
        debug_assert!(self.claimable.is_empty());
        self.claimable.reset();

        let environment = &Environment::new(self, schedule, world);

        pool.scope_with_executor(false, thread_executor, |scope| {
            let context = Context {
                environment,
                scope,
                error_handler,
            };

            // The first tick won't need to process finished systems, but we still need to run the loop in
            // tick_executor() in case a system completes while the first tick still holds the mutex.
            context.tick_executor();
        });

        // End the borrows of self and world in environment by copying out the reference to systems.
        let systems = environment.systems;

        debug_assert_eq!(self.active_workers.0.load(Ordering::Relaxed), 0);
        debug_assert!(self.claimable.is_empty());

        let state = self.state.get_mut().unwrap();
        if self.apply_final_deferred {
            // Do one final apply buffers after all systems have completed
            // Commands should be applied while on the scope's thread, not the executor's thread
            let res = apply_deferred(&state.unapplied_systems, systems, world);
            if let Err(payload) = res {
                let panic_payload = self.panic_payload.get_mut().unwrap();
                *panic_payload = Some(payload);
            }
            state.unapplied_systems.clear();
        }

        // check to see if there was a panic
        let payload = self.panic_payload.get_mut().unwrap();
        if let Some(payload) = payload.take() {
            std::panic::resume_unwind(payload);
        }

        debug_assert!(state.ready_systems.is_clear());
        debug_assert!(state.running_systems.is_clear());
        state.evaluated_sets.clear();
        state.skipped_systems.clear();
        state.completed_systems.clear();
    }

    fn set_apply_final_deferred(&mut self, value: bool) {
        self.apply_final_deferred = value;
    }
}

impl<'scope, 'env: 'scope, 'sys> Context<'scope, 'env, 'sys> {
    fn system_completed(
        &self,
        system_index: usize,
        res: Result<(), Box<dyn Any + Send>>,
        system: &ScheduleSystem,
    ) {
        if let Err(payload) = res {
            #[cfg(feature = "std")]
            #[expect(clippy::print_stderr, reason = "Allowed behind `std` feature gate.")]
            {
                eprintln!("Encountered a panic in system `{}`!", system.name());
            }
            // set the payload to propagate the error
            {
                let mut panic_payload = self.environment.executor.panic_payload.lock().unwrap();
                *panic_payload = Some(payload);
            }
        }
        // Fast path: the lock is free, so handle this completion (and any queued ones)
        // directly without going through the completion queue.
        if let Some((conditions, mut guard)) = self.try_lock() {
            guard.finish_system_and_handle_dependents(SystemResult { system_index });
            let drained = guard.tick(self, conditions);
            drop(guard);
            if TRACK_IN_FLIGHT {
                self.environment
                    .executor
                    .in_flight
                    .0
                    .fetch_sub(1 + drained, Ordering::Relaxed);
            }
            if self.environment.executor.system_completion.is_empty() {
                return;
            }
            self.tick_executor();
            return;
        }
        // Slow path: another thread is scheduling. Hand the completion over; that thread
        // re-checks the queue after releasing the lock, so it cannot be missed.
        self.environment
            .executor
            .system_completion
            .push(SystemResult { system_index })
            .unwrap_or_else(|error| unreachable!("{}", error));
        self.tick_executor();
    }

    #[expect(
        clippy::mut_from_ref,
        reason = "Field is only accessed here and is guarded by lock with a documented safety comment"
    )]
    fn try_lock<'a>(&'a self) -> Option<(&'a mut Conditions<'sys>, MutexGuard<'a, ExecutorState>)> {
        let guard = self.environment.executor.state.try_lock().ok()?;
        // SAFETY: This is an exclusive access as no other location fetches conditions mutably, and
        // is synchronized by the lock on the executor state.
        let conditions = unsafe { &mut *self.environment.conditions.get() };
        Some((conditions, guard))
    }

    fn tick_executor(&self) {
        // Ensure that the executor handles any events pushed to the system_completion queue by this thread.
        // If this thread acquires the lock, the executor runs after the push() and they are processed.
        // If this thread does not acquire the lock, then the is_empty() check on the other thread runs
        // after the lock is released, which is after try_lock() failed, which is after the push()
        // on this thread, so the is_empty() check will see the new events and loop.
        loop {
            let Some((conditions, mut guard)) = self.try_lock() else {
                return;
            };
            let drained = guard.tick(self, conditions);
            // Make sure we drop the guard before checking system_completion.is_empty(), or we could lose events.
            drop(guard);
            if TRACK_IN_FLIGHT {
                self.environment
                    .executor
                    .in_flight
                    .0
                    .fetch_sub(drained, Ordering::Relaxed);
            }
            if self.environment.executor.system_completion.is_empty() {
                return;
            }
        }
    }

    /// The body of a worker task: runs claimable systems until there are none left.
    fn worker(&self) {
        let executor = self.environment.executor;
        let mut done: ArrayVec<usize, MAX_CLAIM> = ArrayVec::new();
        loop {
            let range = match self.claim_batch() {
                Some(range) => range,
                None => match self.linger() {
                    Some(range) => range,
                    None => {
                        // Retire. The dispatcher counts live workers to decide whether to spawn
                        // new ones, so re-check the queue after decrementing: either it observes
                        // the decrement and spawns a replacement, or we observe its push and
                        // stay alive.
                        executor.active_workers.0.fetch_sub(1, Ordering::SeqCst);
                        fence(Ordering::SeqCst);
                        if executor.claimable.is_empty() {
                            return;
                        }
                        executor.active_workers.0.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }
                },
            };
            for slot in range {
                let system_index = executor.claimable.get(slot);
                self.run_one(system_index);
                done.push(system_index);
                // Fast path: the lock is free, so report everything finished so far and
                // dispatch whatever that releases.
                if FAST_PATH_COMPLETION && let Some((conditions, mut guard)) = self.try_lock() {
                    for &index in &done {
                        guard.finish_system_and_handle_dependents(SystemResult {
                            system_index: index,
                        });
                    }
                    let finished = done.len();
                    done.clear();
                    let drained = guard.tick(self, conditions);
                    drop(guard);
                    if TRACK_IN_FLIGHT {
                        executor
                            .in_flight
                            .0
                            .fetch_sub(finished + drained, Ordering::Relaxed);
                    }
                    if !executor.system_completion.is_empty() {
                        self.tick_executor();
                    }
                }
            }
            if !done.is_empty() {
                // Slow path: another thread is scheduling. Hand the completions over; that
                // thread re-checks the queue after releasing the lock, so they cannot be missed.
                for &index in &done {
                    executor
                        .system_completion
                        .push(SystemResult {
                            system_index: index,
                        })
                        .unwrap_or_else(|error| unreachable!("{}", error));
                }
                done.clear();
                self.tick_executor();
            }
        }
    }

    /// Claims the next system (see [`MAX_CLAIM`]) with one read of the queue's head and tail
    /// and one compare-and-swap; the same reads tell whether work is still queued behind it.
    fn claim_batch(&self) -> Option<core::ops::Range<usize>> {
        let executor = self.environment.executor;
        let (range, remaining) = executor.claimable.claim(MAX_CLAIM)?;
        // Ramp up tree-style: if work is still queued behind this claim, add one more worker
        // (it will do the same), instead of the dispatcher spawning one task per queued
        // system up front, most of which would find the queue already drained.
        if TREE_SPAWN && remaining > 0 {
            self.spawn_worker_if_below_max();
        }
        Some(range)
    }

    /// Spawns one more worker task unless the pool is already saturated. Racy on purpose:
    /// an occasional extra worker just retires again.
    fn spawn_worker_if_below_max(&self) {
        let executor = self.environment.executor;
        if executor.active_workers.0.load(Ordering::Relaxed) >= executor.max_workers {
            return;
        }
        executor.active_workers.0.fetch_add(1, Ordering::SeqCst);
        let context = *self;
        self.scope.spawn(async move {
            context.worker();
        });
    }

    /// Nothing is claimable right now. If systems are still running, their completions are
    /// likely to release more work within microseconds, so keep polling for a short while
    /// instead of retiring and being respawned for the next phase.
    fn linger(&self) -> Option<core::ops::Range<usize>> {
        let executor = self.environment.executor;
        if executor.in_flight.0.load(Ordering::Relaxed) == 0 {
            return None;
        }
        if WORKER_LINGER.is_zero() {
            return None;
        }
        let start = Instant::now();
        loop {
            if let Some(range) = self.claim_batch() {
                return Some(range);
            }
            if executor.in_flight.0.load(Ordering::Relaxed) == 0 {
                return None;
            }
            // Checked on every poll: the budget is small and a late exit here delays the
            // end of the whole schedule run.
            if start.elapsed() >= WORKER_LINGER {
                return None;
            }
            // A short fixed pause between polls keeps a crowd of idle workers from hammering
            // the claim queue's cache lines under the workers still claiming.
            for _ in 0..16 {
                core::hint::spin_loop();
            }
        }
    }

    /// Runs one claimed system, recording a panic if it raises one.
    fn run_one(&self, system_index: usize) {
        // SAFETY: the dispatcher marked this system as running while holding the lock,
        // so it is not borrowed anywhere else; it stays that way until we report completion.
        let system = &mut unsafe { &mut *self.environment.systems[system_index].get() }.system;

        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY:
            // - The dispatcher verified that no running system conflicts with this one,
            //   so we have permission to access the world data used by the system.
            // - `is_exclusive` returned false for every claimable system.
            unsafe {
                if let Err(RunSystemError::Failed(err)) =
                    __rust_begin_short_backtrace::run_unsafe(system, self.environment.world_cell)
                {
                    (self.error_handler)(
                        err,
                        ErrorContext::System {
                            name: system.name(),
                            last_run: system.get_last_run(),
                        },
                    );
                }
            };
        }));
        if let Err(payload) = res {
            #[cfg(feature = "std")]
            #[expect(clippy::print_stderr, reason = "Allowed behind `std` feature gate.")]
            {
                eprintln!("Encountered a panic in system `{}`!", system.name());
            }
            let mut panic_payload = self.environment.executor.panic_payload.lock().unwrap();
            *panic_payload = Some(payload);
        }
    }
}

impl MultiThreadedExecutor {
    /// Creates a new `multi_threaded` executor for use with a [`Schedule`].
    ///
    /// [`Schedule`]: crate::schedule::Schedule
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ExecutorState::new()),
            system_completion: ConcurrentQueue::unbounded(),
            claimable: ClaimQueue::with_capacity(0),
            active_workers: Padded(AtomicUsize::new(0)),
            in_flight: Padded(AtomicUsize::new(0)),
            max_workers: 1,
            starting_systems: FixedBitSet::new(),
            apply_final_deferred: true,
            panic_payload: Mutex::new(None),
            #[cfg(feature = "trace")]
            executor_span: info_span!("multithreaded executor"),
        }
    }
}

impl ExecutorState {
    fn new() -> Self {
        Self {
            system_task_metadata: Vec::new(),
            set_condition_conflicting_systems: Vec::new(),
            num_running_systems: 0,
            num_dependencies_remaining: Vec::new(),
            local_thread_running: false,
            exclusive_running: false,
            evaluated_sets: FixedBitSet::new(),
            ready_systems: FixedBitSet::new(),
            ready_systems_copy: FixedBitSet::new(),
            running_systems: FixedBitSet::new(),
            skipped_systems: FixedBitSet::new(),
            completed_systems: FixedBitSet::new(),
            unapplied_systems: FixedBitSet::new(),
            unapplied_scratch: Vec::new(),
        }
    }

    /// Processes queued completions and dispatches whatever became ready. Returns how many
    /// completions were drained; the caller subtracts them (plus any it processed itself)
    /// from `in_flight` only after this returns, so that the dependents are already
    /// claimable by the time lingering workers might see the count drop.
    fn tick(&mut self, context: &Context, conditions: &mut Conditions) -> usize {
        #[cfg(feature = "trace")]
        let _span = context.environment.executor.executor_span.enter();

        let mut drained = 0;
        for result in context.environment.executor.system_completion.try_iter() {
            self.finish_system_and_handle_dependents(result);
            drained += 1;
        }

        // SAFETY:
        // - `finish_system_and_handle_dependents` has updated the currently running systems.
        // - `rebuild_active_access` locks access for all currently running systems.
        unsafe {
            self.spawn_system_tasks(context, conditions);
        }
        drained
    }

    /// # Safety
    /// - Caller must ensure that `self.ready_systems` does not contain any systems that
    ///   have been mutably borrowed (such as the systems currently running).
    /// - `world_cell` must have permission to access all world data (not counting
    ///   any world data that is claimed by systems currently running on this executor).
    unsafe fn spawn_system_tasks(&mut self, context: &Context, conditions: &mut Conditions) {
        if self.exclusive_running {
            return;
        }
        if self.ready_systems.is_clear() {
            self.spawn_workers(context);
            return;
        }

        #[cfg(feature = "hotpatching")]
        #[expect(
            clippy::undocumented_unsafe_blocks,
            reason = "This actually could result in UB if a system tries to mutate
            `HotPatchChanges`. We allow this as the resource only exists with the `hotpatching` feature.
            and `hotpatching` should never be enabled in release."
        )]
        #[cfg(feature = "hotpatching")]
        let hotpatch_tick = unsafe {
            context
                .environment
                .world_cell
                .get_resource_ref::<HotPatchChanges>()
        }
        .map(|r| r.last_changed())
        .unwrap_or_default();

        // can't borrow since loop mutably borrows `self`
        let mut ready_systems = core::mem::take(&mut self.ready_systems_copy);

        // Skipping systems may cause their dependents to become ready immediately.
        // If that happens, we need to run again immediately or we may fail to spawn those dependents.
        let mut check_for_new_ready_systems = true;
        while check_for_new_ready_systems {
            check_for_new_ready_systems = false;

            ready_systems.clone_from(&self.ready_systems);

            for system_index in ready_systems.ones() {
                debug_assert!(!self.running_systems.contains(system_index));
                // SAFETY: Caller assured that these systems are not running.
                // Therefore, no other reference to this system exists and there is no aliasing.
                let system =
                    &mut unsafe { &mut *context.environment.systems[system_index].get() }.system;

                #[cfg(feature = "hotpatching")]
                if hotpatch_tick.is_newer_than(
                    system.get_last_run(),
                    context.environment.world_cell.change_tick(),
                ) {
                    system.refresh_hotpatch();
                }

                if !self.can_run(system_index, conditions) {
                    // NOTE: exclusive systems with ambiguities are susceptible to
                    // being significantly displaced here (compared to single-threaded order)
                    // if systems after them in topological order can run
                    // if that becomes an issue, `break;` if exclusive system
                    continue;
                }

                self.ready_systems.remove(system_index);

                // SAFETY: `can_run` returned true, which means that:
                // - There can be no systems running whose accesses would conflict with any conditions.
                if unsafe {
                    !self.should_run(
                        system_index,
                        system,
                        conditions,
                        context.environment.world_cell,
                        context.error_handler,
                    )
                } {
                    self.skip_system_and_signal_dependents(system_index);
                    // signal_dependents may have set more systems to ready.
                    check_for_new_ready_systems = true;
                    continue;
                }

                self.running_systems.insert(system_index);
                self.num_running_systems += 1;
                if TRACK_IN_FLIGHT {
                    context
                        .environment
                        .executor
                        .in_flight
                        .0
                        .fetch_add(1, Ordering::Relaxed);
                }

                let system_meta = &self.system_task_metadata[system_index];

                if system_meta.is_exclusive {
                    // SAFETY: `can_run` returned true for this system,
                    // which means no systems are currently borrowed.
                    unsafe {
                        self.spawn_exclusive_system_task(context, system_index);
                    }
                    check_for_new_ready_systems = false;
                    break;
                }

                if !system_meta.is_send {
                    // SAFETY:
                    // - Caller ensured no other reference to this system exists.
                    // - `system_task_metadata[system_index].is_exclusive` is `false`,
                    //   so `System::is_exclusive` returned `false` when we called it.
                    // - `can_run` returned true, so no systems with conflicting world access are running.
                    unsafe {
                        self.spawn_local_system_task(context, system_index);
                    }
                    continue;
                }

                // Hand the system to a worker. The queue holds one slot per system and each
                // system is pushed at most once per run, so it never fills up.
                context.environment.executor.claimable.push(system_index);
            }
        }

        // give back
        self.ready_systems_copy = ready_systems;

        self.spawn_workers(context);
    }

    /// Makes sure enough workers are alive to drain the claimable queue.
    fn spawn_workers(&self, context: &Context) {
        let executor = context.environment.executor;
        let pending = executor.claimable.len();
        if pending == 0 {
            return;
        }
        // Pair with the fence in `Context::worker`: a retiring worker either sees our pushes
        // or we see its decrement, so claimable systems can never be left without a worker.
        fence(Ordering::SeqCst);
        let alive = executor.active_workers.0.load(Ordering::SeqCst);
        // Seed at most two workers here; workers that find more queued work spawn further
        // ones themselves (see `Context::worker`).
        let wanted = pending
            .min(executor.max_workers)
            .saturating_sub(alive)
            .min(DISPATCH_SEED_WORKERS);
        for _ in 0..wanted {
            executor.active_workers.0.fetch_add(1, Ordering::SeqCst);
            let context = *context;
            context.scope.spawn(async move {
                context.worker();
            });
        }
    }

    fn can_run(&mut self, system_index: usize, conditions: &mut Conditions) -> bool {
        // Nothing is running, so nothing can conflict.
        if self.num_running_systems == 0 {
            return true;
        }

        let system_meta = &self.system_task_metadata[system_index];
        if system_meta.is_exclusive {
            return false;
        }

        if !system_meta.is_send && self.local_thread_running {
            return false;
        }

        // TODO: an earlier out if world's archetypes did not change
        for set_idx in conditions.sets_with_conditions_of_systems[system_index]
            .difference(&self.evaluated_sets)
        {
            if !self.set_condition_conflicting_systems[set_idx].is_disjoint(&self.running_systems) {
                return false;
            }
        }

        if !system_meta
            .condition_conflicting_systems
            .is_disjoint(&self.running_systems)
        {
            return false;
        }

        if !self.skipped_systems.contains(system_index)
            && !system_meta
                .conflicting_systems
                .is_disjoint(&self.running_systems)
        {
            return false;
        }

        true
    }

    /// # Safety
    /// * `world` must have permission to read any world data required by
    ///   the system's conditions: this includes conditions for the system
    ///   itself, and conditions for any of the system's sets.
    unsafe fn should_run(
        &mut self,
        system_index: usize,
        system: &mut ScheduleSystem,
        conditions: &mut Conditions,
        world: UnsafeWorldCell,
        error_handler: ErrorHandler,
    ) -> bool {
        let mut should_run = !self.skipped_systems.contains(system_index);

        for set_idx in conditions.sets_with_conditions_of_systems[system_index].ones() {
            if self.evaluated_sets.contains(set_idx) {
                continue;
            }

            // Evaluate the system set's conditions.
            // SAFETY:
            // - The caller ensures that `world` has permission to read any data
            //   required by the conditions.
            let set_conditions_met = unsafe {
                evaluate_and_fold_conditions(
                    &mut conditions.set_conditions[set_idx],
                    world,
                    error_handler,
                    system,
                    true,
                )
            };

            if !set_conditions_met {
                self.skipped_systems
                    .union_with(&conditions.systems_in_sets_with_conditions[set_idx]);
            }

            should_run &= set_conditions_met;
            self.evaluated_sets.insert(set_idx);
        }

        // Evaluate the system's conditions.
        // SAFETY:
        // - The caller ensures that `world` has permission to read any data
        //   required by the conditions.
        let system_conditions_met = unsafe {
            evaluate_and_fold_conditions(
                &mut conditions.system_conditions[system_index],
                world,
                error_handler,
                system,
                false,
            )
        };

        if !system_conditions_met {
            self.skipped_systems.insert(system_index);
        }

        should_run &= system_conditions_met;

        should_run
    }

    /// Spawns a task for a system that needs `!Send` data, on the main thread.
    ///
    /// # Safety
    /// - Caller must not alias systems that are running.
    /// - `is_exclusive` must have returned `false` for the specified system.
    /// - `world` must have permission to access the world data
    ///   used by the specified system.
    unsafe fn spawn_local_system_task(&mut self, context: &Context, system_index: usize) {
        // SAFETY: this system is not running, no other reference exists
        let system = &mut unsafe { &mut *context.environment.systems[system_index].get() }.system;
        // Move the full context object into the new future.
        let context = *context;

        let task = async move {
            let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
                // SAFETY:
                // - The caller ensures that we have permission to
                // access the world data used by the system.
                // - `is_exclusive` returned false
                unsafe {
                    if let Err(RunSystemError::Failed(err)) =
                        __rust_begin_short_backtrace::run_unsafe(
                            system,
                            context.environment.world_cell,
                        )
                    {
                        (context.error_handler)(
                            err,
                            ErrorContext::System {
                                name: system.name(),
                                last_run: system.get_last_run(),
                            },
                        );
                    }
                };
            }));
            context.system_completed(system_index, res, system);
        };

        self.local_thread_running = true;
        context.scope.spawn_on_external(task);
    }

    /// # Safety
    /// Caller must ensure no systems are currently borrowed.
    unsafe fn spawn_exclusive_system_task(&mut self, context: &Context, system_index: usize) {
        // SAFETY: this system is not running, no other reference exists
        let system = &mut unsafe { &mut *context.environment.systems[system_index].get() }.system;
        // Move the full context object into the new future.
        let context = *context;

        if is_apply_deferred(&**system) {
            // Hand the set of unapplied systems to the task without allocating: swap in a
            // spare bitset (or an empty one on the first use, which allocates once).
            let mut unapplied_systems = self.unapplied_scratch.pop().unwrap_or_default();
            unapplied_systems.clear();
            unapplied_systems.grow(self.unapplied_systems.len());
            core::mem::swap(&mut unapplied_systems, &mut self.unapplied_systems);
            let task = async move {
                // SAFETY: `can_run` returned true for this system, which means
                // that no other systems currently have access to the world.
                let world = unsafe { context.environment.world_cell.world_mut() };
                let res = apply_deferred(&unapplied_systems, context.environment.systems, world);
                // Return the bitset for reuse. Nothing else can hold the lock for long right now
                // (an exclusive system is running), but never block on it.
                let mut unapplied_systems = unapplied_systems;
                if let Ok(mut state) = context.environment.executor.state.try_lock() {
                    unapplied_systems.clear();
                    state.unapplied_scratch.push(unapplied_systems);
                }
                context.system_completed(system_index, res, system);
            };

            context.scope.spawn_on_scope(task);
        } else {
            let task = async move {
                // SAFETY: `can_run` returned true for this system, which means
                // that no other systems currently have access to the world.
                let world = unsafe { context.environment.world_cell.world_mut() };
                let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    if let Err(RunSystemError::Failed(err)) =
                        __rust_begin_short_backtrace::run(system, world)
                    {
                        (context.error_handler)(
                            err,
                            ErrorContext::System {
                                name: system.name(),
                                last_run: system.get_last_run(),
                            },
                        );
                    }
                }));
                context.system_completed(system_index, res, system);
            };

            context.scope.spawn_on_scope(task);
        }

        self.exclusive_running = true;
        self.local_thread_running = true;
    }

    fn finish_system_and_handle_dependents(&mut self, result: SystemResult) {
        let SystemResult { system_index, .. } = result;
        // `in_flight` is decremented by the caller (see `tick`), which has the executor.

        if self.system_task_metadata[system_index].is_exclusive {
            self.exclusive_running = false;
        }

        if !self.system_task_metadata[system_index].is_send {
            self.local_thread_running = false;
        }

        debug_assert!(self.num_running_systems >= 1);
        self.num_running_systems -= 1;
        self.running_systems.remove(system_index);
        self.completed_systems.insert(system_index);
        self.unapplied_systems.insert(system_index);

        self.signal_dependents(system_index);
    }

    fn skip_system_and_signal_dependents(&mut self, system_index: usize) {
        self.completed_systems.insert(system_index);
        self.signal_dependents(system_index);
    }

    fn signal_dependents(&mut self, system_index: usize) {
        for &dep_idx in &self.system_task_metadata[system_index].dependents {
            let remaining = &mut self.num_dependencies_remaining[dep_idx];
            debug_assert!(*remaining >= 1);
            *remaining -= 1;
            if *remaining == 0 && !self.completed_systems.contains(dep_idx) {
                self.ready_systems.insert(dep_idx);
            }
        }
    }
}

fn apply_deferred(
    unapplied_systems: &FixedBitSet,
    systems: &[SyncUnsafeCell<SystemWithAccess>],
    world: &mut World,
) -> Result<(), Box<dyn Any + Send>> {
    for system_index in unapplied_systems.ones() {
        // SAFETY: none of these systems are running, no other references exist
        let system = &mut unsafe { &mut *systems[system_index].get() }.system;
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            system.apply_deferred(world);
        }));
        if let Err(payload) = res {
            #[cfg(feature = "std")]
            #[expect(clippy::print_stderr, reason = "Allowed behind `std` feature gate.")]
            {
                eprintln!(
                    "Encountered a panic when applying buffers for system `{}`!",
                    system.name()
                );
            }
            return Err(payload);
        }
    }
    Ok(())
}

/// # Safety
/// - `world` must have permission to read any world data
///   required by `conditions`.
unsafe fn evaluate_and_fold_conditions(
    conditions: &mut [ConditionWithAccess],
    world: UnsafeWorldCell,
    error_handler: ErrorHandler,
    for_system: &ScheduleSystem,
    on_set: bool,
) -> bool {
    #[expect(
        clippy::unnecessary_fold,
        reason = "Short-circuiting here would prevent conditions from mutating their own state as needed."
    )]
    conditions
        .iter_mut()
        .map(|ConditionWithAccess { condition, .. }| {
            // SAFETY:
            // - The caller ensures that `world` has permission to read any data
            //   required by the condition.
            unsafe { __rust_begin_short_backtrace::readonly_run_unsafe(&mut **condition, world) }
                .unwrap_or_else(|err| {
                    if let RunSystemError::Failed(err) = err {
                        error_handler(
                            err,
                            ErrorContext::RunCondition {
                                name: condition.name(),
                                last_run: condition.get_last_run(),
                                system: for_system.name(),
                                on_set,
                            },
                        );
                    };
                    false
                })
        })
        .fold(true, |acc, res| acc && res)
}

/// New-typed [`ThreadExecutor`] [`Resource`] that is used to run systems on the main thread
#[derive(Resource, Clone)]
pub struct MainThreadExecutor(pub Arc<ThreadExecutor<'static>>);

impl Default for MainThreadExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl MainThreadExecutor {
    /// Creates a new executor that can be used to run systems on the main thread.
    pub fn new() -> Self {
        MainThreadExecutor(TaskPool::get_thread_executor())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        prelude::{ResMut, Resource, SystemSet},
        schedule::{IntoScheduleConfigs, MultiThreadedExecutor, Schedule},
        system::{Commands, NonSendMarker},
        world::World,
    };
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Resource)]
    struct R;

    #[test]
    fn skipped_systems_notify_dependents() {
        let mut world = World::new();
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new());
        schedule.add_systems(
            (
                (|| {}).run_if(|| false),
                // This system depends on a system that is always skipped.
                |mut commands: Commands| {
                    commands.insert_resource(R);
                },
            )
                .chain(),
        );
        schedule.run(&mut world);
        assert!(world.get_resource::<R>().is_some());
    }

    /// Regression test for a weird bug flagged by MIRI in
    /// `spawn_exclusive_system_task`, related to a `&mut World` being captured
    /// inside an `async` block and somehow remaining alive even after its last use.
    #[test]
    fn check_spawn_exclusive_system_task_miri() {
        let mut world = World::new();
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new());
        schedule.add_systems(((|_: Commands| {}), |_: Commands| {}).chain());
        schedule.run(&mut world);
    }

    #[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
    struct Link(usize);

    #[derive(Resource, Default)]
    struct Log(Vec<usize>);

    /// A long dependency chain mixed with an exclusive system, an `ApplyDeferred` and a
    /// `!Send` system, run many times: every system must run exactly once per schedule
    /// run and the chain must stay in order.
    #[test]
    fn workers_run_every_system_once_in_order() {
        let mut world = World::new();
        world.init_resource::<Log>();
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new());

        const LINKS: usize = 64;
        for i in 0..LINKS {
            schedule.add_systems((move |mut log: ResMut<Log>| log.0.push(i)).in_set(Link(i)));
            if i > 0 {
                schedule.configure_sets(Link(i).after(Link(i - 1)));
            }
        }
        schedule.add_systems(
            (
                |mut commands: Commands| commands.insert_resource(R),
                |world: &mut World| {
                    assert!(world.contains_resource::<R>());
                    world.resource_mut::<Log>().0.push(1000);
                },
                |_: NonSendMarker, mut log: ResMut<Log>| log.0.push(2000),
            )
                .chain(),
        );
        for _ in 0..200 {
            world.remove_resource::<R>();
            world.resource_mut::<Log>().0.clear();
            schedule.run(&mut world);
            let log = &world.resource::<Log>().0;
            let chain: Vec<usize> = log.iter().copied().filter(|&x| x < LINKS).collect();
            assert_eq!(chain, (0..LINKS).collect::<Vec<_>>());
            assert_eq!(log.iter().filter(|&&x| x == 1000).count(), 1);
            assert_eq!(log.iter().filter(|&&x| x == 2000).count(), 1);
            assert_eq!(log.len(), LINKS + 2);
            assert!(world.get_resource::<R>().is_some());
        }
    }

    /// Many independent systems with no conflicts: the wide, fan-out case.
    #[test]
    fn workers_fan_out() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let mut world = World::new();
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new());
        for _ in 0..500 {
            schedule.add_systems(|| {
                COUNT.fetch_add(1, Ordering::Relaxed);
            });
        }
        for run in 1..=20 {
            schedule.run(&mut world);
            assert_eq!(COUNT.load(Ordering::Relaxed), 500 * run);
        }
    }

    /// Systems that all conflict on one resource: only one may run at a time, and every
    /// one of them must still run.
    #[test]
    fn workers_serialize_conflicting_systems() {
        #[derive(Resource, Default)]
        struct Counter(usize);
        let mut world = World::new();
        world.init_resource::<Counter>();
        let mut schedule = Schedule::default();
        schedule.set_executor(MultiThreadedExecutor::new());
        for _ in 0..200 {
            schedule.add_systems(|mut c: ResMut<Counter>| c.0 += 1);
        }
        for run in 1..=20 {
            schedule.run(&mut world);
            assert_eq!(world.resource::<Counter>().0, 200 * run);
        }
    }
}
