## 3.4.0 — 2026-04-29

### Added
- `BowlInfo.food_type_visible` now accepts `None` (default) for scenes with no
  bowl visible (e.g. non-feeder pet photos in `animal_dog_cat_v1_raw`). Existing
  samples with one of the four literals `dry|wet|mixed|unknown` remain valid;
  this is an additive minor bump.
- Prompt v1.0 system rule #9: 画面无碗时 `bowl.food_fill_ratio /
  water_fill_ratio / food_type_visible` 三字段全部设为 null。
- Few-shot example #7: 无碗户外宠物抓拍 (negative bowl pattern teaches the model
  the canonical null shape).

### Changed
- JSON Schema `bowl.food_type_visible` removed from `required`; type widened to
  `oneOf: [null, string-enum]`.
- `tests/test_pydantic_sync.test_food_type_enum` adapted to the Optional
  annotation (extracts inner Literal arm via `typing.get_args`).

### Motivation
Empirical probe v2 against doubao-seed-2-0-mini-260215: 0/10 schema_valid
on `animal_dog_cat_v1_raw` (10 575 non-feeder photos), 100% failing on
`food_type_visible: null` returned by an honest VLM that doesn't see a bowl.
Schema acceptance + null-canonical prompt closes the gap without sacrificing
the original 4-literal vocabulary for true feeder data.

## 2.4.0 — 2026-04-22

### Added
- `ModelCard.resolved_config_uri: str | None = None` — URI of the per-run resolved
  Hydra config dump, enabling deterministic `pet run --replay <card-id>` (Phase 4 W4).
