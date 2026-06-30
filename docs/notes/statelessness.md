# Statelessness in durable, resumable execution — a design note

A neutral consideration, not a pitch for any tool. When a computation has to
**pause and resume** — wait for a human, survive a crash, continue after a deploy —
you face one architectural choice before any API detail: **where does the paused
run's state live, and what is running while it waits?** Statelessness is one answer
to that question. This note works through its value, its costs, the workarounds,
and the cases where it is the right call.

## The choice, stated plainly

A paused computation's progress has to live *somewhere*. There are two families:

- **Stateful execution.** A live runtime holds the run — in a worker's memory, in a
  background task, in a cluster. The run is *suspended*, not gone; something is
  keeping it alive and will resume it. Durability comes from that runtime persisting
  and recovering its own state.
- **Stateless re-invocation.** Nothing is kept alive. The run's progress is
  externalized to a store as plain data; between pause and resume **no process,
  task, or daemon represents it**. Resuming is a fresh invocation that reconstructs
  where it was — typically by **replaying** completed steps from the store and
  running only what's left.

This is the same dividing line that recurs all over systems design: server-side
sessions vs. self-contained tokens, stateful connections vs. stateless request
protocols, long-lived servers vs. functions that spin up per request. The tradeoffs
rhyme each time, which is why it's worth treating as a pattern rather than a
per-tool detail.

## The value of statelessness

- **Nothing to operate while idle.** A waiting run consumes no process, thread, or
  worker. Ten thousand runs paused for a week cost what their stored rows cost, not
  what ten thousand live workflows cost.
- **Horizontal scale is trivial.** Because no instance *owns* a paused run, any
  instance can resume any run. There is no affinity to preserve, no rebalancing when
  a node joins or leaves, no "which worker has this workflow."
- **Crash safety is cheap by construction.** If nothing is running during the pause,
  a crash during the pause loses nothing — there is no in-flight runtime state to
  reconstruct beyond what's already in the store. Recovery isn't a subsystem; it's
  just the next invocation.
- **One uniform mechanism for "continue later."** Every reason a run resumes — a
  human answering, a transient failure retried, a scheduled wake-up — collapses to
  the *same* primitive: invoke again, replay, proceed. Stateful systems tend to grow
  distinct mechanisms for each (a signal, a retry policy, a durable timer).
- **State is inspectable data.** Progress lives in a store you can query, diff, and
  reason about, rather than inside an opaque running process. Debugging a stuck run
  is reading rows, not attaching to a runtime.
- **Edge/serverless-native.** Environments that discourage long-lived background
  processes are exactly where stateful runtimes are awkward and statelessness is
  natural — the model already matches "spin up, do a unit, return."

## The cost of statelessness

None of the above is free. The same property that removes the runtime removes what
the runtime was doing for you.

- **It cannot wake itself.** With nothing alive during the pause, there is no clock
  inside the system to fire "continue in three days." Autonomous, long-horizon
  resumption has to come from *outside*.
- **Resuming means replaying.** Reconstructing position by replaying completed steps
  imposes a **determinism constraint**: the code between persisted steps must take
  the same path on re-execution, and every nondeterministic or side-effecting step
  must be captured so it is *replayed*, not *re-run*. This is a real discipline, not
  an implementation detail.
- **Per-resume reconstruction has a cost.** Each continuation re-hydrates state and
  replays history. For short runs this is negligible; for runs with **long
  histories** it grows, and at high throughput the repeated reconstruction can
  matter. Stateful runtimes that keep the run warm pay this once.
- **No warm in-memory state across a pause.** Caches, open connections, and
  accumulated in-process context do not survive to the resume. Anything that must
  persist has to be externalized and re-acquired, which can mean cold reconnects.
- **Concurrency must be coordinated externally.** Two requests resuming the same run
  at once can double-run. With no owning process to serialize them, the store has to
  — via a lock, optimistic concurrency, or session-level serialization.
- **State accretion is your problem.** History that enables replay also accumulates.
  Without pruning or snapshotting, the store and the replay cost grow with the run's
  age.

## Workarounds

Most of the costs have well-trodden mitigations; the point is that they are
*explicit, ordinary parts* rather than capabilities a runtime hides.

- **Self-waking → an external trigger.** A scheduler, cron, or delay-queue calls the
  resume endpoint at the due time. Because re-invocation is idempotent (replay makes
  a repeated call safe), this is a thin, debuggable part — and most stacks already
  run a scheduler. You trade a hidden durable timer for a visible "something calls a
  URL." It is a new moving part only for a stack that has *no* scheduler at all.
- **Replay cost → bound the history.** Snapshot/checkpoint to collapse old history,
  cap retained steps, and content-address work so unchanged steps are skipped rather
  than recomputed. Keeps reconstruction roughly constant regardless of age.
- **Determinism → quarantine nondeterminism.** Push every nondeterministic or
  side-effecting action behind a captured boundary whose *result* is what replays.
  The orchestration between those boundaries stays replay-safe by staying pure.
- **Concurrency → externalized serialization.** A per-run lock or optimistic version
  in the store gives single-writer semantics without an owning process.
- **Warm state → externalize and re-acquire.** Move caches/connections to a shared
  store or pool and accept cold acquisition on resume; for hot paths, a separate
  warm tier the stateless flow reads through.
- **Exactly-once side effects → idempotency keys at the boundary.** A replayed step
  that already committed its effect must dedupe (a key the downstream honors),
  rather than relying on the step never re-running.

## When statelessness is the right call

It is a genuine tradeoff, not a default. Statelessness tends to win when:

- **Work is request- or session-scoped** and naturally bounded — a turn, an
  approval, a pipeline that completes in seconds-to-hours, not a process that lives
  for weeks.
- **There are many independent flows** rather than one giant coordinated workflow —
  scale is "more sessions," which the no-affinity property handles for free.
- **The deployment discourages background runtimes** — serverless, edge, or a team
  that does not want to operate a stateful execution tier.
- **Operational simplicity outweighs autonomy** — you would rather have an
  inspectable store and a uniform resume than a runtime that does more but must be
  run, scaled, and recovered.
- **State is naturally externalizable** and histories stay modest, so replay cost is
  not the bottleneck.

It is the **wrong** call when the workload's essence is exactly what the runtime
provides:

- **Long-horizon autonomous waits** — days or weeks sleeping on a timer that must
  fire by itself, with no external caller to lean on.
- **Distributed coordination of a single flow** — steps that must run across
  machines and be orchestrated as one workflow, including cross-service rollbacks and
  compensation.
- **Throughput where per-resume replay is too expensive** — very high volume or very
  long histories, where keeping the run warm amortizes better than reconstructing it.
- **The execution history is itself the product** — strict, queryable audit of every
  step as a first-class requirement.

## The shape of the conclusion

Statelessness does not make a system simpler in the abstract; it **relocates**
complexity — out of a runtime you operate and into explicit parts you can see: a
store, an external trigger, an idempotency key, a determinism discipline. When the
workload is bounded, plural, and externalizable, that relocation is a clear net win:
less to run, trivial scale-out, cheap crash safety, one resume mechanism. When the
workload's value *is* autonomy, distribution, or warm long-lived coordination, the
relocation removes the very thing you needed, and a stateful runtime earns its
operational weight. The real engineering question is never "stateless or not" in
general — it is whether your work lives on the bounded-and-plural side of that line
or the autonomous-and-coordinated side.
