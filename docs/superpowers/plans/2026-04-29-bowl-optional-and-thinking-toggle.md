# Bowl Optional + Thinking Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two coordinated PRs unblock high-throughput annotation of `animal_dog_cat_v1_raw` (10 575 images) on doubao-seed-2.0-mini: (1) make `bowl.food_type_visible` optional in pet-schema so no-bowl scenes pass validation; (2) add per-annotator `extra_payload` passthrough in pet-annotation so we can disable Doubao's reasoning mode at runtime.

**Architecture:** Both PRs are additive (non-breaking minor bumps). pet-schema 3.3.0 → 3.4.0 (Optional + None default on `food_type_visible`). pet-annotation 2.1.1 → 2.2.0 (new optional `extra_payload: dict | None = None` on `LLMAnnotatorConfig`, threaded into the request body in `OpenAICompatProvider._build_payload`). pet-annotation will keep its current pet-schema β peer-dep range; no version pin bump required for the schema change.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, ruff, mypy, conda env `pet-pipeline`. Volcengine ARK chat-completions API for runtime evidence.

**Empirical evidence motivating these PRs (probe v2, 2026-04-29):**

| group | wall (10 imgs) | avg_lat | avg_out tok | schema |
|---|---:|---:|---:|---:|
| baseline (sync, thinking ON) | 400.8s | 40.1s | 4370 | 1/10 |
| thinking OFF (sync) | 63.1s | 6.3s | 537 | 4/10 |
| concurrent N=10, thinking ON | 55.6s | 45.2s | 4947 | 0/10 |
| **concurrent N=10, thinking OFF** | **6.3s** | **5.5s** | **538** | 5/10 |

Schema failures all collapse to one root cause:

```
('bowl', 'food_type_visible') :: literal_error ::
Input should be 'dry', 'wet', 'mixed' or 'unknown'
```

The model returns `null` whenever the scene has no bowl (every image in `animal_dog_cat_v1_raw` is a non-feeder pet photo). Once `food_type_visible` accepts `None`, schema_ok must rise to 10/10.

---

## File Structure

### pet-schema (PR #1)
- **Modify** `src/pet_schema/models.py:131` — `food_type_visible` annotation widens to `Literal[...] | None = None`.
- **Modify** `src/pet_schema/versions/v1.0/prompt_user.jinja2:61` — value spec adds `|null`.
- **Modify** `src/pet_schema/versions/v1.0/prompt_system.txt` — add explicit rule: 画面无碗时 `bowl.food_fill_ratio / water_fill_ratio / food_type_visible` 全设 null.
- **Modify** `src/pet_schema/versions/v1.0/schema.json` — drop `food_type_visible` from required; allow `string | null`.
- **Modify** `src/pet_schema/versions/v1.0/few_shot_examples.json` — add a "no-bowl scene" example so the model sees the null pattern.
- **Modify** `src/pet_schema/version.py` — `SCHEMA_VERSION = "3.4.0"`.
- **Modify** `pyproject.toml` — `version = "3.4.0"`.
- **Modify** `CHANGELOG.md` — prepend a 3.4.0 entry.
- **Create** `tests/test_food_type_optional.py` — new failing test asserting `food_type_visible=None` is accepted by `BowlInfo` and `PetFeederEvent`.
- **Modify** `tests/test_pydantic_sync.py:103-109` — update `test_food_type_enum` to handle the `Literal | None` annotation correctly (extract inner Literal from Union).

### pet-annotation (PR #2)
- **Modify** `src/pet_annotation/config.py` — add `extra_payload: dict[str, Any] | None = None` to `LLMAnnotatorConfig`.
- **Modify** `src/pet_annotation/teacher/providers/openai_compat.py` — `__init__` accepts optional `extra_payload`; `_build_payload` deep-merges it into the dict.
- **Modify** `src/pet_annotation/teacher/orchestrator.py:81-112` — `_build_provider` propagates `llm_cfg.extra_payload` to the constructor.
- **Modify** `params.yaml` — comment in the `llm.annotators` example block showing `extra_payload: {thinking: {type: disabled}}`.
- **Create** `tests/test_extra_payload.py` — failing tests:
  - `LLMAnnotatorConfig` accepts `extra_payload` and validates as dict.
  - `OpenAICompatProvider._build_payload` includes the merged thinking key.
  - `_build_provider` wires `extra_payload` into the constructed provider.

---

## PR #1 — pet-schema v3.4.0 (`feature/v3.4.0-bowl-optional` → `dev` → `main`)

### Task 1: Worktree + branch setup

**Files:** N/A (git plumbing).

- [ ] **Step 1: Sync `dev` and create feature branch**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
git fetch origin
git checkout dev
git pull --ff-only origin dev
git checkout -b feature/v3.4.0-bowl-optional
```

Expected: branch shows `On branch feature/v3.4.0-bowl-optional`, working tree clean (the four `.DS_Store` untracked files seen earlier are git-ignored / harmless — leave them alone).

### Task 2: Failing test for Optional food_type_visible

**Files:**
- Create: `tests/test_food_type_optional.py`

- [ ] **Step 1: Write the failing test**

```python
"""v3.4.0 — bowl.food_type_visible accepts null when no bowl visible.

Schema changes from required Literal to Literal | None = None so that VLM
output for non-feeder scenes (no bowl in frame) passes validation.
"""
from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from pet_schema.models import BowlInfo, PetFeederEvent


def test_bowlinfo_accepts_none_food_type():
    bowl = BowlInfo(food_fill_ratio=None, water_fill_ratio=None,
                    food_type_visible=None)
    assert bowl.food_type_visible is None


def test_bowlinfo_accepts_omitted_food_type():
    bowl = BowlInfo(food_fill_ratio=None, water_fill_ratio=None)
    assert bowl.food_type_visible is None


def test_bowlinfo_still_rejects_unknown_string():
    with pytest.raises(ValidationError):
        BowlInfo(food_fill_ratio=None, water_fill_ratio=None,
                 food_type_visible="kibble")  # not in enum


def test_petfeedeerevent_no_pet_no_bowl():
    """Pet-absent scene with no bowl visible — common in non-feeder datasets."""
    event = PetFeederEvent(
        schema_version="1.0",
        pet_present=False,
        pet_count=0,
        pet=None,
        bowl=BowlInfo(food_fill_ratio=None, water_fill_ratio=None,
                      food_type_visible=None),
        scene={"lighting": "bright", "image_quality": "clear",
               "confidence_overall": 0.5},
        narrative="empty room, no pets visible",
    )
    assert event.bowl.food_type_visible is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema && conda run -n pet-pipeline pytest tests/test_food_type_optional.py -v`
Expected: 3 failures with `food_type_visible literal_error` (one passes — `test_bowlinfo_still_rejects_unknown_string`).

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_food_type_optional.py
git commit -m "test(pet-schema): failing tests for v3.4.0 food_type_visible Optional"
```

### Task 3: Make `food_type_visible` Optional in Pydantic model

**Files:**
- Modify: `src/pet_schema/models.py:131`

- [ ] **Step 1: Edit annotation**

Replace line 131 of `src/pet_schema/models.py`:

```python
# before
    food_type_visible: Literal["dry", "wet", "mixed", "unknown"]

# after
    food_type_visible: Literal["dry", "wet", "mixed", "unknown"] | None = None
```

- [ ] **Step 2: Run new tests to verify they pass**

Run: `conda run -n pet-pipeline pytest tests/test_food_type_optional.py -v`
Expected: 4/4 PASS.

- [ ] **Step 3: Run full schema test suite to confirm no regression**

Run: `conda run -n pet-pipeline pytest -x`
Expected: All pass except possibly `test_pydantic_sync.py::test_food_type_enum` (Task 5 fixes that). If any other test breaks, stop and surface.

### Task 4: Update prompt + JSON schema + few-shot examples in lockstep

**Files:**
- Modify: `src/pet_schema/versions/v1.0/prompt_system.txt`
- Modify: `src/pet_schema/versions/v1.0/prompt_user.jinja2:61`
- Modify: `src/pet_schema/versions/v1.0/schema.json:103-111`
- Modify: `src/pet_schema/versions/v1.0/few_shot_examples.json`

- [ ] **Step 1: Add a no-bowl rule to system prompt**

After line 11 of `prompt_system.txt` (the existing narrative rule), append a new line 12:

```
9. 画面中无碗时（如户外/抓拍/居家无喂食器场景），bowl.food_fill_ratio / water_fill_ratio / food_type_visible 三个字段全部设为 null，禁止编造碗或食物。
```

- [ ] **Step 2: Update user prompt template**

`prompt_user.jinja2` line 61:

```jinja
# before
    "food_type_visible":<"dry"|"wet"|"mixed"|"unknown">

# after
    "food_type_visible":<"dry"|"wet"|"mixed"|"unknown"|null>
```

- [ ] **Step 3: Update JSON schema**

In `schema.json`:
- Line 105: drop `food_type_visible` from `bowl.required` → `"required": []`.
- Line 110: change to `"food_type_visible": {"oneOf": [{"type": "null"}, {"type": "string", "enum": ["dry", "wet", "mixed", "unknown"]}]}`.

- [ ] **Step 4: Add no-bowl few-shot example**

Append a new example object to `few_shot_examples.json` (before the closing `]`):

```json
,
  {
    "scene_desc": "无碗的户外宠物抓拍",
    "output": {
      "schema_version": "1.0",
      "pet_present": true,
      "pet_count": 1,
      "pet": {
        "species": "dog",
        "breed_estimate": "mixed_unknown",
        "id_tag": "brown_medium_dog",
        "id_confidence": 0.55,
        "action": {
          "primary": "other",
          "distribution": {
            "eating": 0.00, "drinking": 0.00,
            "sniffing_only": 0.05, "leaving_bowl": 0.00,
            "sitting_idle": 0.10, "other": 0.85
          }
        },
        "eating_metrics": {
          "speed": {"fast": 0.00, "normal": 0.00, "slow": 0.00},
          "engagement": 0.00,
          "abandoned_midway": 0.00
        },
        "mood": {"alertness": 0.55, "anxiety": 0.10, "engagement": 0.40},
        "body_signals": {"posture": "relaxed", "ear_position": "forward"},
        "anomaly_signals": {
          "vomit_gesture": 0.00, "food_rejection": 0.00,
          "excessive_sniffing": 0.00, "lethargy": 0.00, "aggression": 0.00
        }
      },
      "bowl": {
        "food_fill_ratio": null,
        "water_fill_ratio": null,
        "food_type_visible": null
      },
      "scene": {
        "lighting": "bright",
        "image_quality": "clear",
        "confidence_overall": 0.70
      },
      "narrative": "棕色中型犬在草地上行走，无喂食器及碗具入镜。"
    }
  }
```

- [ ] **Step 5: Run full test suite**

Run: `conda run -n pet-pipeline pytest`
Expected: All pass except `test_pydantic_sync.py::test_food_type_enum` which Task 5 fixes.

- [ ] **Step 6: Commit**

```bash
git add src/pet_schema/models.py src/pet_schema/versions/v1.0/
git commit -m "feat(pet-schema): make bowl.food_type_visible Optional + null rule in prompt"
```

### Task 5: Update `test_pydantic_sync.test_food_type_enum`

**Files:**
- Modify: `tests/test_pydantic_sync.py:103-109`

The current test does `set(annotation.__args__)` directly on the `Literal[...] | None` annotation; that yields `{Literal[...], NoneType}` instead of the literal members. Fix it to extract the Literal arm.

- [ ] **Step 1: Update test**

```python
def test_food_type_enum(self, v1_schema):
    schema_enum = set(
        v1_schema["properties"]["bowl"]["properties"]["food_type_visible"][
            "oneOf"
        ][1]["enum"]
    )
    annotation = BowlInfo.model_fields["food_type_visible"].annotation
    # annotation is `Literal[...] | None`; pull the Literal arm
    import typing
    args = typing.get_args(annotation)
    literal_args = [a for a in args if a is not type(None)]
    assert len(literal_args) == 1
    pydantic_enum = set(typing.get_args(literal_args[0]))
    assert schema_enum == pydantic_enum
```

- [ ] **Step 2: Run the specific test**

Run: `conda run -n pet-pipeline pytest tests/test_pydantic_sync.py::TestPydanticSync::test_food_type_enum -v`
Expected: PASS.

- [ ] **Step 3: Run full schema suite**

Run: `conda run -n pet-pipeline pytest`
Expected: All pass (167+).

- [ ] **Step 4: Commit**

```bash
git add tests/test_pydantic_sync.py
git commit -m "test(pet-schema): adapt test_food_type_enum to Optional annotation"
```

### Task 6: Version bump + CHANGELOG

**Files:**
- Modify: `src/pet_schema/version.py`
- Modify: `pyproject.toml:9`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version constants**

```python
# src/pet_schema/version.py
SCHEMA_VERSION = "3.4.0"
```

```toml
# pyproject.toml line 9
version = "3.4.0"
```

- [ ] **Step 2: Prepend CHANGELOG entry**

Insert at the top of `CHANGELOG.md`:

```markdown
## 3.4.0 — 2026-04-29

### Added
- `BowlInfo.food_type_visible` now accepts `None` (default) for scenes
  with no bowl visible (e.g. non-feeder pet photos). Existing samples with
  one of the four literals remain valid; this is an additive minor bump.
- Prompt v1.0 system rule #9: 画面无碗时 `bowl.*` 三字段全部设 null。
- Few-shot example: 无碗户外抓拍 (negative bowl pattern).

### Changed
- JSON Schema `bowl.food_type_visible` removed from required; type widened
  to `null | string`.
- `tests/test_pydantic_sync.test_food_type_enum` adapted to Optional
  annotation (extracts inner Literal).

```

- [ ] **Step 3: Verify pyproject parity test still passes**

Run: `conda run -n pet-pipeline pytest tests/ -k version`
Expected: PASS (the pyproject↔SCHEMA_VERSION parity test from v3.0.0).

- [ ] **Step 4: Commit**

```bash
git add src/pet_schema/version.py pyproject.toml CHANGELOG.md
git commit -m "chore(pet-schema): bump 3.3.0 → 3.4.0"
```

### Task 7: Lint + final test pass + push

- [ ] **Step 1: Lint and type-check**

Run: `conda run -n pet-pipeline ruff check src tests && conda run -n pet-pipeline mypy src`
Expected: zero errors. If ruff fixes are needed, run `ruff check --fix` and amend Task 6's commit (only ruff cosmetic fixes; do NOT --amend if commits already pushed).

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feature/v3.4.0-bowl-optional
gh pr create --base dev --title "feat(pet-schema): v3.4.0 bowl.food_type_visible Optional" --body "$(cat <<'EOF'
## Summary
- Make `BowlInfo.food_type_visible` Optional (None default) so VLM output for no-bowl scenes (e.g. `animal_dog_cat_v1_raw` general pet photos) passes schema validation
- Update prompt v1.0 system rule + user template + JSON schema + few-shot example to teach the model the null pattern
- Bump SCHEMA_VERSION 3.3.0 → 3.4.0 (additive, non-breaking)

## Motivation
Empirical probe on doubao-seed-2.0-mini against `animal_dog_cat_v1_raw` (non-feeder pet photos): 0/10 schema_valid, all failing on `food_type_visible: null` returned by an honest VLM that doesn't see a bowl. Probe data: see `_quality_probe/probe_v2.py` results (commit message in branch).

## Test plan
- [ ] `pytest` green (167+ tests)
- [ ] `ruff check` + `mypy` green
- [ ] Re-run probe v2 with updated schema → expect schema_ok = 10/10
EOF
)"
```

- [ ] **Step 3: Wait for CI green + 1-2 reviewers**

Memory `feedback_pr_workflow`: pet-schema requires 2 reviewers. In single-developer mode, the user merges manually after CI green.

- [ ] **Step 4: After merge, sync `main`**

```bash
git checkout dev
git pull origin dev
gh pr create --base main --head dev --title "sync(pet-schema): dev → main — v3.4.0" --body "Sync v3.4.0 (food_type_visible Optional) to main."
```

- [ ] **Step 5: Tag the release once main is updated**

```bash
git checkout main
git pull origin main
git tag -a v3.4.0 -m "pet-schema v3.4.0 — bowl.food_type_visible Optional"
git push origin v3.4.0
```

- [ ] **Step 6: Sync dev with main (memory `feedback_dev_main_divergence`)**

```bash
git checkout dev
git merge origin/main --no-edit
git push origin dev
```

---

## PR #2 — pet-annotation `extra_payload` (`feature/llm-extra-payload` → `dev` → `main`)

### Task 8: Branch setup

- [ ] **Step 1: Sync and branch**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-annotation
git fetch origin
git checkout dev
git pull --ff-only origin dev
git checkout -b feature/llm-extra-payload
```

### Task 9: Failing tests for `extra_payload`

**Files:**
- Create: `tests/test_extra_payload.py`

- [ ] **Step 1: Write failing tests**

```python
"""v2.2.0 — LLMAnnotatorConfig.extra_payload threads opaque kwargs into VLM request body.

Used to disable Doubao's reasoning mode (thinking={"type":"disabled"}) at runtime.
"""
import pytest

from pet_annotation.config import LLMAnnotatorConfig
from pet_annotation.teacher.orchestrator import _build_provider
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


def test_llm_annotator_config_accepts_extra_payload():
    cfg = LLMAnnotatorConfig(
        id="doubao-seed-2-0-mini",
        provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model_name="doubao-seed-2-0-mini-260215",
        extra_payload={"thinking": {"type": "disabled"}},
    )
    assert cfg.extra_payload == {"thinking": {"type": "disabled"}}


def test_extra_payload_default_is_none():
    cfg = LLMAnnotatorConfig(
        id="x", provider="openai_compat",
        base_url="https://example.com", model_name="m",
    )
    assert cfg.extra_payload is None


def test_provider_build_payload_merges_extra(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG header
    p = OpenAICompatProvider(
        base_url="https://example.com/v1",
        model_name="m",
        extra_payload={"thinking": {"type": "disabled"}},
    )
    payload = p._build_payload(str(img), ("sys", "user"))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.1


def test_provider_no_extra_payload_no_thinking_key(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    p = OpenAICompatProvider(base_url="https://example.com/v1", model_name="m")
    payload = p._build_payload(str(img), ("sys", "user"))
    assert "thinking" not in payload


def test_build_provider_propagates_extra_payload():
    cfg = LLMAnnotatorConfig(
        id="d", provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model_name="doubao-seed-2-0-mini-260215",
        extra_payload={"thinking": {"type": "disabled"}},
    )
    provider = _build_provider(cfg)
    assert provider._extra_payload == {"thinking": {"type": "disabled"}}
```

- [ ] **Step 2: Run failing tests**

Run: `conda run -n pet-pipeline pytest tests/test_extra_payload.py -v`
Expected: all 5 fail (attribute / field not present).

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_extra_payload.py
git commit -m "test(pet-annotation): failing tests for extra_payload passthrough"
```

### Task 10: Add `extra_payload` to LLMAnnotatorConfig

**Files:**
- Modify: `src/pet_annotation/config.py`

- [ ] **Step 1: Add field**

In `LLMAnnotatorConfig` (after `api_key: str = ""`):

```python
    api_key: str = ""
    extra_payload: dict[str, Any] | None = None
    """Opaque key/value pairs deep-merged into the chat-completions request body.

    Use to enable provider-specific knobs that are not in the OpenAI shape, e.g.
    ``{"thinking": {"type": "disabled"}}`` to disable Doubao reasoning. Keys here
    take precedence over the provider's default body. Default: None (no merge).
    """
```

(Verify `Any` is in the existing `from typing import` block at the top of the file; if not, add it.)

- [ ] **Step 2: Run config tests**

Run: `conda run -n pet-pipeline pytest tests/test_extra_payload.py::test_llm_annotator_config_accepts_extra_payload tests/test_extra_payload.py::test_extra_payload_default_is_none -v`
Expected: 2 PASS, 3 still fail.

### Task 11: Thread `extra_payload` through `OpenAICompatProvider`

**Files:**
- Modify: `src/pet_annotation/teacher/providers/openai_compat.py`

- [ ] **Step 1: Update `__init__` signature and store**

In `OpenAICompatProvider.__init__` after `max_tokens: int = 2048,`:

```python
        max_tokens: int = 2048,
        extra_payload: dict | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_payload = extra_payload or {}
        self._session: aiohttp.ClientSession | None = None
```

- [ ] **Step 2: Merge into `_build_payload`**

At the end of `_build_payload` (just before `return`):

```python
        result = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._extra_payload:
            result.update(self._extra_payload)
        return result
```

(Replace the existing `return {...}`.)

- [ ] **Step 3: Run provider tests**

Run: `conda run -n pet-pipeline pytest tests/test_extra_payload.py::test_provider_build_payload_merges_extra tests/test_extra_payload.py::test_provider_no_extra_payload_no_thinking_key -v`
Expected: 2 PASS.

### Task 12: Wire `_build_provider` to pass `extra_payload`

**Files:**
- Modify: `src/pet_annotation/teacher/orchestrator.py:81-112`

- [ ] **Step 1: Update each branch in `_build_provider`**

For each of the three branches (`vllm`, `doubao`, `openai_compat`), add `extra_payload=llm_cfg.extra_payload`. Example for the doubao branch:

```python
    if llm_cfg.provider == "doubao":
        return DoubaoProvider(
            base_url=llm_cfg.base_url,
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            extra_payload=llm_cfg.extra_payload,
        )
```

Apply analogously to `openai_compat` and `vllm` branches. (`VLLMProvider` extends `OpenAICompatProvider`; if its `__init__` overrides, mirror the kwarg there too — check `providers/vllm.py` and either inherit cleanly or thread through.)

- [ ] **Step 2: Run last failing test**

Run: `conda run -n pet-pipeline pytest tests/test_extra_payload.py -v`
Expected: 5/5 PASS.

- [ ] **Step 3: Run full pet-annotation suite**

Run: `conda run -n pet-pipeline pytest`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/pet_annotation/config.py src/pet_annotation/teacher/providers/openai_compat.py src/pet_annotation/teacher/orchestrator.py
git commit -m "feat(pet-annotation): LLMAnnotatorConfig.extra_payload passthrough for thinking toggle"
```

### Task 13: params.yaml example + version bump + CHANGELOG

**Files:**
- Modify: `params.yaml`
- Modify: `pyproject.toml` version
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add commented example to params.yaml**

In the `llm.annotators` example block, add an `extra_payload` line:

```yaml
  # annotators:
  #   - id: "doubao-seed-2-0-mini"
  #     provider: "doubao"
  #     base_url: "https://ark.cn-beijing.volces.com/api/v3"
  #     model_name: "doubao-seed-2-0-mini-260215"
  #     temperature: 0.1
  #     max_tokens: 2048
  #     api_key: ""
  #     extra_payload:
  #       thinking: {type: disabled}  # disable Doubao reasoning mode
```

- [ ] **Step 2: Bump version**

In `pet-annotation/pyproject.toml`, bump `version` from `2.1.1` → `2.2.0` (verify exact current value first).

- [ ] **Step 3: Prepend CHANGELOG entry**

```markdown
## 2.2.0 — 2026-04-29

### Added
- `LLMAnnotatorConfig.extra_payload: dict | None` — opaque kwargs deep-merged
  into the chat-completions request body. Used to disable Doubao reasoning
  mode via `extra_payload: {thinking: {type: disabled}}` (≈6× speedup on
  doubao-seed-2.0-mini per probe v2 2026-04-29).
- `OpenAICompatProvider` constructor accepts `extra_payload`; `_build_payload`
  merges it.
- `_build_provider` propagates `extra_payload` through all three provider
  branches (openai_compat / doubao / vllm).
```

- [ ] **Step 4: Lint + tests**

```bash
conda run -n pet-pipeline ruff check src tests
conda run -n pet-pipeline mypy src
conda run -n pet-pipeline pytest
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add params.yaml pyproject.toml CHANGELOG.md
git commit -m "chore(pet-annotation): bump 2.1.1 → 2.2.0 + params.yaml example"
```

### Task 14: Push + PR + sync to main + tag

- [ ] **Step 1: Push and open PR**

```bash
git push -u origin feature/llm-extra-payload
gh pr create --base dev --title "feat(pet-annotation): v2.2.0 LLMAnnotatorConfig.extra_payload passthrough" --body "$(cat <<'EOF'
## Summary
- New `LLMAnnotatorConfig.extra_payload: dict | None = None` field
- `OpenAICompatProvider._build_payload` deep-merges this into chat-completions body
- Threaded through `_build_provider` for all three providers (openai_compat / doubao / vllm)
- Use case: `extra_payload: {thinking: {type: disabled}}` disables Doubao reasoning mode → 6× speedup, 8× fewer output tokens

## Motivation
Probe v2 against doubao-seed-2.0-mini-260215: thinking off changes avg per-image latency 40s → 6s and avg completion tokens 4370 → 537. With concurrency N=10, full annotation of `animal_dog_cat_v1_raw` (10 575 images) goes from ~96h to ~1.5–3h.

## Test plan
- [ ] `pytest tests/test_extra_payload.py -v` — 5/5 PASS
- [ ] full `pytest` green
- [ ] ruff + mypy green
- [ ] no breaking changes (default `None` ⇒ existing behavior)
EOF
)"
```

- [ ] **Step 2: After CI green + merge, sync to main and tag**

```bash
git checkout dev
git pull origin dev
gh pr create --base main --head dev --title "sync(pet-annotation): dev → main — v2.2.0" --body "Sync v2.2.0 (extra_payload passthrough) to main."
# After merge:
git checkout main && git pull origin main
git tag -a v2.2.0 -m "pet-annotation v2.2.0 — extra_payload passthrough"
git push origin v2.2.0
git checkout dev && git merge origin/main --no-edit && git push origin dev
```

---

## Post-merge verification — re-run probe

### Task 15: Reinstall pet-schema in editable mode (already editable, but verify)

- [ ] **Step 1: Verify version**

```bash
conda run -n pet-pipeline python -c "import pet_schema; print(pet_schema.SCHEMA_VERSION if hasattr(pet_schema,'SCHEMA_VERSION') else 'check')"
```

Expected: `3.4.0`. If still 3.3.0, run `conda run -n pet-pipeline pip install -e /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema`.

### Task 16: Re-run probe v2 (4 groups × 10 images)

- [ ] **Step 1: Probe**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/animal_dog_cat_v1_raw/_quality_probe
conda run -n pet-pipeline --no-capture-output python probe_v2.py \
  --api-key ark-f3173424-6e46-48cd-a743-55415fbc4808-fbb12 --n 10 --seed 42
```

Expected (key acceptance criteria):
- `concurrent N=10 (thinking OFF)` schema_ok = **10/10**
- All other groups schema_ok = 10/10 too (model returns `null` consistently and now schema accepts it)
- wall ≈ 6s for the concurrent + thinking-off group

If schema_ok < 10/10, dump the failure and fix before declaring done.

---

## Final acceptance gate

The two PRs are complete when **all** are true:

- [ ] pet-schema v3.4.0 tagged on `main`, `dev` synced from `main`.
- [ ] pet-annotation v2.2.0 tagged on `main`, `dev` synced from `main`.
- [ ] Probe v2 shows schema_ok = 10/10 in concurrent + thinking-off group.
- [ ] Memory updated (project_pet_schema_status, project_pet_annotation_status).
- [ ] User signed off on launching the full 10 575-image annotation run.
