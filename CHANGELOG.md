# Changelog

All notable changes to Altum are documented in this file.

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

[0.1.0]: https://github.com/Diogenesoftoronto/axon/releases/tag/v0.1.0
