# Changelog

All notable changes to Altum are documented in this file.

## [0.2.0] - 2026-07-11

### Added

- A runtime-neutral sandbox interface with the existing Ouros adapter and an optional sandboxed Steel runtime behind the `steel` feature.
- Deterministic journal replay for suspending and resuming host calls from Steel.
- A pluggable asynchronous VFS interface with memory and host-filesystem backends.
- An optional content-addressed S3/R2/MinIO VFS backend behind the `s3` feature.
- Focused regression coverage for runtime sessions, Scheme repair, VFS isolation, and optional-feature builds.

### Changed

- Filesystem VFS handles now allocate namespaces exclusively, preventing stale state reuse across processes or restarts.
- The release workflow now uses supported macOS 15 Intel and Apple Silicon runners.
- Release examples now reference the canonical `Diogenesoftoronto/axon` repository.

### Fixed

- Reject parent path components before resolving filesystem-backed VFS paths.
- Preserve leading slashes when filtering filesystem VFS list prefixes.
- Protect Altum's logical `main` Ouros session from destruction.
- Treat Scheme apostrophes as quote syntax and track escaped double quotes relative to their position during repair.
- Restore compilation for the advertised `steel` and `s3` Cargo features.

## [0.1.0] - 2026-06-19

### Added

- Forkable Ouros sandbox sessions with checkpoint restore and isolated per-fork VFS workspaces.
- Model-visible checkpoint, fork, VFS, and strategy-commit helpers.
- Optional JSONL trajectory recording with token and recursion telemetry through `--trace-output`.
- `export-sft` for converting recorded trajectories into OpenAI messages-format training data.
- End-to-end trajectory recording and SFT export coverage, plus broader parser, storage, sandbox, and tool-registry tests.

### Changed

- Renamed remaining Axon and PlanCraft benchmark references to Altum.
- Refreshed benchmark scripts, analysis notebooks, paper sources, and supporting documentation.
- Updated the fork, checkpoint, and VFS document from a proposal to the implemented design.
- Refreshed locked Ouros/Ruff dependencies.

### Fixed

- Record completed trajectories on normal and fallback completion paths.
- Include fallback completion usage, generated code, and sandbox observations in trajectory data.
- Make SFT message truncation Unicode-safe.
- Correct RFC 3339 timestamps at month boundaries.

[0.2.0]: https://github.com/Diogenesoftoronto/axon/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Diogenesoftoronto/axon/releases/tag/v0.1.0
