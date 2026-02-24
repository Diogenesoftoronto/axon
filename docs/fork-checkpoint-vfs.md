# Fork, Checkpoint, and VFS Extension (Potential Feature)

This document proposes a workflow extension for Axon where the agent can:

1. Fork reasoning into multiple candidate paths.
2. Checkpoint state at meaningful milestones.
3. Stage and revisit work through a model-visible virtual filesystem (VFS).
4. Commit to a final strategy only after evaluating candidates.

The goal is to improve reliability on hard tasks without forcing one linear chain of thought.

## Why This Extension

Current flow is iteration-based and mostly linear. If the model picks a weak strategy early, recovery is expensive.

A fork and checkpoint flow allows Axon to:

- explore alternatives in parallel or sequence,
- compare outcomes using explicit criteria,
- return to a known-good checkpoint,
- avoid destructive rewrites of intermediate work.

The VFS adds a structured scratchpad where candidate plans, evidence, and partial artifacts can be stored and revisited.

## Proposed Concepts

### Checkpoint

A checkpoint is a restorable snapshot of runtime state at one moment.

Suggested checkpoint payload:

- sandbox state reference (or replay journal boundary),
- current message window id,
- selected variables summary,
- metadata (timestamp, depth, label).

### Fork

A fork is a child branch created from a checkpoint.

Each fork has:

- `fork_id`,
- `parent_checkpoint_id`,
- isolated sandbox execution path,
- own evaluation metrics.

### VFS Workspace

The VFS is a logical filesystem visible to the model as a workspace, not host FS access.

Suggested virtual paths:

- `/plans/` for strategy drafts,
- `/evidence/` for extracted facts and citations,
- `/artifacts/` for generated code or text,
- `/scores/` for branch evaluation notes.

The model can write, read, diff, and promote files across forks through controlled APIs.

## Workflow

1. Create baseline checkpoint after initial recon.
2. Create N forks from baseline (for example: regex-heavy, semantic-heavy, hybrid).
3. In each fork:
   - run sandbox code,
   - call `llm_query` as needed,
   - write intermediate outputs to VFS (`/plans`, `/evidence`, `/artifacts`).
4. Score each fork on explicit criteria:
   - correctness confidence,
   - evidence coverage,
   - cost (iterations/tokens),
   - execution risk.
5. Select best fork.
6. Optionally merge selected artifacts into main branch VFS.
7. Commit final strategy and continue to final answer.

This defers commitment until enough evidence exists.

## API Shape (Draft)

Potential external calls exposed in sandbox:

- `CHECKPOINT_CREATE(label: str) -> checkpoint_id`
- `CHECKPOINT_RESTORE(checkpoint_id: str) -> status`
- `FORK_CREATE(checkpoint_id: str, name: str) -> fork_id`
- `FORK_SWITCH(fork_id: str) -> status`
- `FORK_LIST() -> list`
- `FORK_SCORE(fork_id: str, score_json: str) -> status`
- `VFS_WRITE(path: str, content: str) -> status`
- `VFS_READ(path: str) -> str`
- `VFS_LIST(path: str) -> list`
- `VFS_DIFF(path_a: str, path_b: str) -> str`
- `VFS_PROMOTE(from_path: str, to_path: str) -> status`

Finalization helpers:

- `STRATEGY_COMMIT(fork_id: str, rationale: str) -> status`
- `STRATEGY_STATUS() -> json`

## Implementation Plan

### Phase 1: Minimal Replay-Based Checkpoints

Use execution journaling, not full sandbox snapshotting.

- Persist executed code blocks and external call returns.
- Checkpoint stores journal offset.
- Restore replays journal to rebuild state.

Pros:

- simple and deterministic if externals are memoized,
- no deep ouros internals required.

Cons:

- restore cost grows with journal size.

### Phase 2: Fork Manager

Add a branch table in Rust runtime:

- `fork_id -> {checkpoint, journal_tail, metadata, score}`.

Switching forks updates active journal and active VFS namespace.

### Phase 3: VFS Layer

Implement in-memory first, optional persisted backend later.

- path normalization and size limits,
- per-fork namespaces,
- optional read-only shared base namespace.

### Phase 4: Prompt and Policy Updates

Update prompts to encourage:

- explicit branch naming,
- explicit scorecards,
- strategy commit before final answer.

## Integration Points in Current Codebase

- `src/rlm.rs`
  - execution loop orchestration,
  - external function dispatch (`handle_external`),
  - checkpoint/fork control plane.
- `src/sandbox.rs`
  - register new external names with ouros.
- `src/prompts.rs`
  - root/sub prompts for fork and commit discipline.
- `src/store.rs`
  - optional persistence of VFS and fork metadata.
- `src/mcp.rs`
  - optional tool endpoints for branch inspection/debugging.

## Safety and Limits

Recommended guardrails:

- max forks per depth (for example 3),
- max checkpoint count per query,
- VFS quotas per fork,
- hard timeout per fork execution slice,
- deterministic memoization for external calls during replay.

## Observability

Trace additions to support debugging:

- active fork id in logs,
- checkpoint create/restore events,
- VFS write/read metrics,
- scorecard summary before strategy commit.

This can extend the existing `--trace-sandbox` mode.

## Open Questions

1. Should fork exploration happen only at depth 0, or at all recursion depths?
2. Should VFS be fully isolated per depth, or inherit from parent depth with copy-on-write?
3. What is the default scoring rubric, and should users customize it?
4. Should strategy commit be mandatory before `FINAL(...)`?

## Suggested First Milestone

Deliver a minimal vertical slice:

1. Replay-based `CHECKPOINT_CREATE` and `CHECKPOINT_RESTORE`.
2. Two forks max.
3. In-memory VFS with `VFS_WRITE`, `VFS_READ`, `VFS_LIST`.
4. Prompt update that asks model to compare forks and call `STRATEGY_COMMIT`.
5. Trace output showing fork/checkpoint transitions.

This gives immediate value without major architectural risk.
