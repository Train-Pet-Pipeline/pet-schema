## 2.4.0 — 2026-04-22

### Added
- `ModelCard.resolved_config_uri: str | None = None` — URI of the per-run resolved
  Hydra config dump, enabling deterministic `pet run --replay <card-id>` (Phase 4 W4).
