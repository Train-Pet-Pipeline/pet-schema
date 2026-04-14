# pet-schema v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pet-schema v1.0 package from scratch — Schema definitions, validator, renderer, Pydantic models, and full test suite with anti-drift protection.

**Architecture:** JSON Schema as authoritative source, Pydantic v2 models as internal API, four-direction sync tests to prevent drift. TDD throughout — tests first, then implementation.

**Tech Stack:** Python 3.11, Pydantic v2, jsonschema, Jinja2, pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-04-14-pet-schema-v1-design.md`

---

## File Structure

```
pet-schema/
├── versions/
│   └── v1.0/
│       ├── prompt_system.txt          # system prompt 纯文本
│       ├── prompt_user.jinja2         # user prompt Jinja2 模板
│       ├── schema.json                # JSON Schema Draft-7（权威来源）
│       ├── few_shot_examples.json     # 3 个示例（新格式）
│       └── CHANGELOG.md
├── src/
│   └── pet_schema/
│       ├── __init__.py                # 暴露 validate_output, render_prompt
│       ├── validator.py               # jsonschema + 额外约束验证
│       ├── renderer.py                # Jinja2 渲染 prompt
│       └── models.py                  # Pydantic v2 模型
├── tests/
│   ├── conftest.py                    # 共享 fixtures（示例数据）
│   ├── test_validator.py
│   ├── test_renderer.py
│   ├── test_examples.py
│   └── test_pydantic_sync.py          # 四方向双向一致性校验
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
└── .gitignore
```

---

### Task 1: Project scaffolding — git init, pyproject.toml, Makefile

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/pet_schema/__init__.py` (empty placeholder)

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
.mypy_cache/
.ruff_cache/
.env
.venv/
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pet-schema"
version = "1.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "jsonschema>=4.20,<5.0",
    "pydantic>=2.0,<3.0",
    "jinja2>=3.1,<4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff", "mypy", "pip-tools"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create Makefile**

```makefile
.PHONY: setup test lint clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check . && mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
```

- [ ] **Step 5: Create .env.example**

```
# pet-schema 不需要环境变量，此文件按仓库规范保留
```

- [ ] **Step 6: Create empty package placeholder**

```bash
mkdir -p src/pet_schema
touch src/pet_schema/__init__.py
```

- [ ] **Step 7: Install dev dependencies and verify**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
make setup
```

Expected: 安装成功，无报错

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml Makefile .env.example src/pet_schema/__init__.py
git commit -m "feat(pet-schema): init project scaffolding with pyproject.toml and Makefile"
```

---

### Task 2: Schema definition files — schema.json, prompts, few_shot_examples, CHANGELOG

**Files:**
- Create: `versions/v1.0/schema.json`
- Create: `versions/v1.0/prompt_system.txt`
- Create: `versions/v1.0/prompt_user.jinja2`
- Create: `versions/v1.0/few_shot_examples.json`
- Create: `versions/v1.0/CHANGELOG.md`

- [ ] **Step 1: Create schema.json**

JSON Schema Draft-7，按照 spec 的字段定义。关键变更（vs DEVELOPMENT_GUIDE 原版）：
- `body_signals.posture`: enum 而非 distribution object
- `body_signals.ear_position`: enum 而非 distribution object
- `mood`: 只有 alertness/anxiety/engagement，无 comfort
- `narrative.maxLength`: 80

完整 schema.json 内容：

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PetFeederEvent",
  "version": "1.0",
  "type": "object",
  "required": ["schema_version", "pet_present", "pet_count", "bowl", "scene", "narrative"],
  "properties": {
    "schema_version": {"type": "string", "enum": ["1.0"]},
    "pet_present": {"type": "boolean"},
    "pet_count": {"type": "integer", "minimum": 0, "maximum": 4},
    "pet": {
      "oneOf": [
        {"type": "null"},
        {
          "type": "object",
          "required": ["species", "breed_estimate", "id_tag", "id_confidence",
                       "action", "eating_metrics", "mood", "body_signals", "anomaly_signals"],
          "properties": {
            "species": {"type": "string", "enum": ["cat", "dog", "unknown"]},
            "breed_estimate": {"type": "string", "minLength": 1},
            "id_tag": {"type": "string", "minLength": 1},
            "id_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "action": {
              "type": "object",
              "required": ["primary", "distribution"],
              "properties": {
                "primary": {
                  "type": "string",
                  "enum": ["eating", "drinking", "sniffing_only",
                           "leaving_bowl", "sitting_idle", "other"]
                },
                "distribution": {
                  "type": "object",
                  "required": ["eating", "drinking", "sniffing_only",
                               "leaving_bowl", "sitting_idle", "other"],
                  "properties": {
                    "eating": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "drinking": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "sniffing_only": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "leaving_bowl": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "sitting_idle": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "other": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                  },
                  "additionalProperties": false
                }
              }
            },
            "eating_metrics": {
              "type": "object",
              "required": ["speed", "engagement", "abandoned_midway"],
              "properties": {
                "speed": {
                  "type": "object",
                  "required": ["fast", "normal", "slow"],
                  "properties": {
                    "fast": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "normal": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "slow": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                  },
                  "additionalProperties": false
                },
                "engagement": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "abandoned_midway": {"type": "number", "minimum": 0.0, "maximum": 1.0}
              }
            },
            "mood": {
              "type": "object",
              "required": ["alertness", "anxiety", "engagement"],
              "properties": {
                "alertness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "anxiety": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "engagement": {"type": "number", "minimum": 0.0, "maximum": 1.0}
              }
            },
            "body_signals": {
              "type": "object",
              "required": ["posture", "ear_position"],
              "properties": {
                "posture": {
                  "type": "string",
                  "enum": ["relaxed", "tense", "hunched", "unobservable"]
                },
                "ear_position": {
                  "type": "string",
                  "enum": ["forward", "flat", "rotating", "unobservable"]
                }
              }
            },
            "anomaly_signals": {
              "type": "object",
              "required": ["vomit_gesture", "food_rejection", "excessive_sniffing",
                           "lethargy", "aggression"],
              "properties": {
                "vomit_gesture": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "food_rejection": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "excessive_sniffing": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "lethargy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "aggression": {"type": "number", "minimum": 0.0, "maximum": 1.0}
              }
            }
          }
        }
      ]
    },
    "bowl": {
      "type": "object",
      "required": ["food_type_visible"],
      "properties": {
        "food_fill_ratio": {"oneOf": [{"type": "null"}, {"type": "number", "minimum": 0.0, "maximum": 1.0}]},
        "water_fill_ratio": {"oneOf": [{"type": "null"}, {"type": "number", "minimum": 0.0, "maximum": 1.0}]},
        "food_type_visible": {"type": "string", "enum": ["dry", "wet", "mixed", "unknown"]}
      }
    },
    "scene": {
      "type": "object",
      "required": ["lighting", "image_quality", "confidence_overall"],
      "properties": {
        "lighting": {"type": "string", "enum": ["bright", "dim", "infrared_night"]},
        "image_quality": {"type": "string", "enum": ["clear", "blurry", "partially_occluded"]},
        "confidence_overall": {"type": "number", "minimum": 0.0, "maximum": 1.0}
      }
    },
    "narrative": {"type": "string", "maxLength": 80}
  }
}
```

- [ ] **Step 2: Create prompt_system.txt**

基于 DEVELOPMENT_GUIDE 的 system prompt，调整：
- narrative 限制改为 80 字
- body_signals 说明改为枚举（不再是分布）
- mood 说明去掉 comfort

```
你是一个专业的宠物行为分析系统，负责分析喂食器摄像头拍摄的图像帧。

摄像头参数：固定俯角，对准食碗区域，视角固定不变。
你的任务：对当前帧中的宠物行为、状态和碗的情况进行精确分析。

输出规则（必须严格遵守，违反任何一条均为无效输出）：
1. 只输出 JSON，不输出任何其他内容，不加 markdown 代码块，不加注释
2. 所有概率字段值域为 [0.00, 1.00]，保留两位小数
3. action.distribution 中所有值之和必须为 1.00（允许浮点误差 ±0.01）
4. eating_metrics.speed 中所有值之和必须为 1.00（允许浮点误差 ±0.01）
5. 如果画面中没有宠物，pet_present 设为 false，pet 字段设为 null
6. narrative 只描述当前帧可见内容，不推断历史，不拟人化，不超过 80 字
7. body_signals 的 posture 和 ear_position 为枚举值，无法判断时使用 unobservable
8. 不确定时用低置信度表达，不要给出虚假的高置信度
```

- [ ] **Step 3: Create prompt_user.jinja2**

基于 DEVELOPMENT_GUIDE 模板，按新 schema 调整 body_signals（枚举）、mood（3 维）、narrative（80 字）。

```jinja2
[图像帧]

请分析这张喂食器摄像头图像，严格按以下 Schema 输出 JSON：

{
  "schema_version": "1.0",
  "pet_present": <true|false>,
  "pet_count": <0-4 的整数>,

  "pet": {
    "species": <"cat"|"dog"|"unknown">,
    "breed_estimate": <品种描述字符串，如 "british_shorthair"，不确定写 "mixed_unknown">,
    "id_tag": <用外观特征描述，如 "orange_tabby_large"，用于同一设备上区分多宠>,
    "id_confidence": <0.00-1.00>,

    "action": {
      "primary": <distribution 中概率最高的动作标签>,
      "distribution": {
        "eating":       <0.00-1.00>,
        "drinking":     <0.00-1.00>,
        "sniffing_only":<0.00-1.00>,
        "leaving_bowl": <0.00-1.00>,
        "sitting_idle": <0.00-1.00>,
        "other":        <0.00-1.00>
      }
    },

    "eating_metrics": {
      "speed": {
        "fast":   <0.00-1.00>,
        "normal": <0.00-1.00>,
        "slow":   <0.00-1.00>
      },
      "engagement":      <0.00-1.00，1=全神贯注进食>,
      "abandoned_midway":<0.00-1.00，1=明确中途放弃>
    },

    "mood": {
      "alertness":  <0.00-1.00，1=高度警觉>,
      "anxiety":    <0.00-1.00，1=明显焦虑>,
      "engagement": <0.00-1.00，1=高度投入>
    },

    "body_signals": {
      "posture": <"relaxed"|"tense"|"hunched"|"unobservable">,
      "ear_position": <"forward"|"flat"|"rotating"|"unobservable">
    },

    "anomaly_signals": {
      "vomit_gesture":      <0.00-1.00，检测到呕吐前姿态序列>,
      "food_rejection":     <0.00-1.00，嗅闻后明确拒绝进食>,
      "excessive_sniffing": <0.00-1.00，异常长时间嗅闻不进食，>30秒>,
      "lethargy":           <0.00-1.00，动作迟缓无力，相比正常明显减慢>,
      "aggression":         <0.00-1.00，攻击性行为>
    }
  },

  "bowl": {
    "food_fill_ratio":  <0.00-1.00，0=空碗，1=满碗；无法判断写 null>,
    "water_fill_ratio": <0.00-1.00；无水碗写 null>,
    "food_type_visible":<"dry"|"wet"|"mixed"|"unknown">
  },

  "scene": {
    "lighting":           <"bright"|"dim"|"infrared_night">,
    "image_quality":      <"clear"|"blurry"|"partially_occluded">,
    "confidence_overall": <0.00-1.00，本次整体分析置信度>
  },

  "narrative": "<80字以内，客观描述当前可见行为，不拟人化>"
}

{% if few_shot_examples %}
参考示例（格式参考，不要复制内容）：
{{ few_shot_examples }}
{% endif %}
```

- [ ] **Step 4: Create few_shot_examples.json**

3 个示例，按新 schema 格式（枚举 body_signals、3 维 mood、80 字 narrative）：

```json
[
  {
    "scene_desc": "白天，正常进食",
    "output": {
      "schema_version": "1.0",
      "pet_present": true,
      "pet_count": 1,
      "pet": {
        "species": "cat",
        "breed_estimate": "british_shorthair",
        "id_tag": "grey_shorthair_medium",
        "id_confidence": 0.83,
        "action": {
          "primary": "eating",
          "distribution": {
            "eating": 0.76, "drinking": 0.00,
            "sniffing_only": 0.14, "leaving_bowl": 0.05,
            "sitting_idle": 0.03, "other": 0.02
          }
        },
        "eating_metrics": {
          "speed": {"fast": 0.08, "normal": 0.71, "slow": 0.21},
          "engagement": 0.74,
          "abandoned_midway": 0.12
        },
        "mood": {
          "alertness": 0.28,
          "anxiety": 0.09,
          "engagement": 0.76
        },
        "body_signals": {
          "posture": "relaxed",
          "ear_position": "forward"
        },
        "anomaly_signals": {
          "vomit_gesture": 0.02, "food_rejection": 0.09,
          "excessive_sniffing": 0.16, "lethargy": 0.04, "aggression": 0.01
        }
      },
      "bowl": {
        "food_fill_ratio": 0.42,
        "water_fill_ratio": null,
        "food_type_visible": "dry"
      },
      "scene": {
        "lighting": "bright",
        "image_quality": "clear",
        "confidence_overall": 0.85
      },
      "narrative": "灰色英短以正常速度进食干粮，碗内余粮约42%，状态放松，偶尔抬头观察。"
    }
  },
  {
    "scene_desc": "夜视红外，无宠物",
    "output": {
      "schema_version": "1.0",
      "pet_present": false,
      "pet_count": 0,
      "pet": null,
      "bowl": {
        "food_fill_ratio": 0.88,
        "water_fill_ratio": null,
        "food_type_visible": "dry"
      },
      "scene": {
        "lighting": "infrared_night",
        "image_quality": "clear",
        "confidence_overall": 0.92
      },
      "narrative": "无宠物，碗内余粮充足约88%，夜视模式。"
    }
  },
  {
    "scene_desc": "挑食拒食，视觉特征部分遮挡",
    "output": {
      "schema_version": "1.0",
      "pet_present": true,
      "pet_count": 1,
      "pet": {
        "species": "cat",
        "breed_estimate": "orange_tabby",
        "id_tag": "orange_tabby_large",
        "id_confidence": 0.71,
        "action": {
          "primary": "sniffing_only",
          "distribution": {
            "eating": 0.08, "drinking": 0.00,
            "sniffing_only": 0.67, "leaving_bowl": 0.18,
            "sitting_idle": 0.05, "other": 0.02
          }
        },
        "eating_metrics": {
          "speed": {"fast": 0.02, "normal": 0.15, "slow": 0.83},
          "engagement": 0.31,
          "abandoned_midway": 0.74
        },
        "mood": {
          "alertness": 0.52,
          "anxiety": 0.41,
          "engagement": 0.28
        },
        "body_signals": {
          "posture": "tense",
          "ear_position": "unobservable"
        },
        "anomaly_signals": {
          "vomit_gesture": 0.06, "food_rejection": 0.78,
          "excessive_sniffing": 0.71, "lethargy": 0.12, "aggression": 0.03
        }
      },
      "bowl": {
        "food_fill_ratio": 0.91,
        "water_fill_ratio": null,
        "food_type_visible": "wet"
      },
      "scene": {
        "lighting": "dim",
        "image_quality": "partially_occluded",
        "confidence_overall": 0.62
      },
      "narrative": "橘猫反复嗅闻湿粮后未进食，身体略微紧绷，有明显拒食迹象，碗内余量约91%。"
    }
  }
]
```

- [ ] **Step 5: Create CHANGELOG.md**

```markdown
# CHANGELOG - Schema v1.0

## v1.0 (2026-04-14)

初始版本。

### 与 DEVELOPMENT_GUIDE 原版的差异

- `body_signals.posture`: 从 3 值概率分布改为枚举（含 unobservable）
- `body_signals.ear_position`: 从 4 值概率分布改为枚举（含 unobservable）
- `mood.comfort`: 删除（与 anxiety 负相关冗余）
- `narrative.maxLength`: 从 50 放宽到 80
```

- [ ] **Step 6: Commit**

```bash
git add versions/
git commit -m "feat(pet-schema): add v1.0 schema definition, prompts, and few-shot examples"
```

---

### Task 3: Pydantic v2 models — models.py (TDD)

**Files:**
- Create: `src/pet_schema/models.py`
- Create: `tests/conftest.py`
- Create: `tests/test_examples.py` (partial — Pydantic parsing tests)

- [ ] **Step 1: Create conftest.py with shared fixtures**

```python
"""Shared test fixtures for pet-schema tests."""
import json
from pathlib import Path

import pytest

VERSIONS_DIR = Path(__file__).parent.parent / "versions"


@pytest.fixture
def v1_examples() -> list[dict]:
    """Load v1.0 few-shot examples."""
    path = VERSIONS_DIR / "v1.0" / "few_shot_examples.json"
    return json.loads(path.read_text())


@pytest.fixture
def v1_schema() -> dict:
    """Load v1.0 JSON Schema."""
    path = VERSIONS_DIR / "v1.0" / "schema.json"
    return json.loads(path.read_text())


@pytest.fixture
def valid_eating_output() -> dict:
    """A valid eating event for testing."""
    return {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "breed_estimate": "british_shorthair",
            "id_tag": "grey_shorthair_medium",
            "id_confidence": 0.83,
            "action": {
                "primary": "eating",
                "distribution": {
                    "eating": 0.76, "drinking": 0.00,
                    "sniffing_only": 0.14, "leaving_bowl": 0.05,
                    "sitting_idle": 0.03, "other": 0.02,
                },
            },
            "eating_metrics": {
                "speed": {"fast": 0.08, "normal": 0.71, "slow": 0.21},
                "engagement": 0.74,
                "abandoned_midway": 0.12,
            },
            "mood": {"alertness": 0.28, "anxiety": 0.09, "engagement": 0.76},
            "body_signals": {"posture": "relaxed", "ear_position": "forward"},
            "anomaly_signals": {
                "vomit_gesture": 0.02, "food_rejection": 0.09,
                "excessive_sniffing": 0.16, "lethargy": 0.04, "aggression": 0.01,
            },
        },
        "bowl": {
            "food_fill_ratio": 0.42,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "bright",
            "image_quality": "clear",
            "confidence_overall": 0.85,
        },
        "narrative": "灰色英短以正常速度进食干粮，碗内余粮约42%，状态放松。",
    }


@pytest.fixture
def valid_no_pet_output() -> dict:
    """A valid no-pet event for testing."""
    return {
        "schema_version": "1.0",
        "pet_present": False,
        "pet_count": 0,
        "pet": None,
        "bowl": {
            "food_fill_ratio": 0.88,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "infrared_night",
            "image_quality": "clear",
            "confidence_overall": 0.92,
        },
        "narrative": "无宠物，碗内余粮充足约88%，夜视模式。",
    }
```

- [ ] **Step 2: Write failing tests for Pydantic models**

```python
# tests/test_models.py — 临时测试文件，后续 test_examples.py 和 test_pydantic_sync.py 会覆盖更多
"""Tests for Pydantic v2 models."""
import pytest
from pydantic import ValidationError

from pet_schema.models import PetFeederEvent


class TestPetFeederEventParsing:
    """Test that Pydantic models can parse valid data."""

    def test_parse_valid_eating(self, valid_eating_output):
        event = PetFeederEvent.model_validate(valid_eating_output)
        assert event.pet_present is True
        assert event.pet is not None
        assert event.pet.action.primary == "eating"
        assert event.pet.body_signals.posture == "relaxed"
        assert event.pet.mood.alertness == 0.28

    def test_parse_valid_no_pet(self, valid_no_pet_output):
        event = PetFeederEvent.model_validate(valid_no_pet_output)
        assert event.pet_present is False
        assert event.pet is None

    def test_action_distribution_sum_valid(self, valid_eating_output):
        """Distribution summing to 1.0 should pass."""
        event = PetFeederEvent.model_validate(valid_eating_output)
        dist = event.pet.action.distribution
        total = dist.eating + dist.drinking + dist.sniffing_only + dist.leaving_bowl + dist.sitting_idle + dist.other
        assert abs(total - 1.0) <= 0.01

    def test_action_distribution_sum_invalid(self, valid_eating_output):
        """Distribution not summing to 1.0 should fail."""
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["action"] = {**data["pet"]["action"]}
        data["pet"]["action"]["distribution"] = {
            "eating": 0.90, "drinking": 0.90,
            "sniffing_only": 0.00, "leaving_bowl": 0.00,
            "sitting_idle": 0.00, "other": 0.00,
        }
        with pytest.raises(ValidationError, match="distribution"):
            PetFeederEvent.model_validate(data)

    def test_speed_distribution_sum_invalid(self, valid_eating_output):
        """Speed not summing to 1.0 should fail."""
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["eating_metrics"] = {**data["pet"]["eating_metrics"]}
        data["pet"]["eating_metrics"]["speed"] = {"fast": 0.90, "normal": 0.90, "slow": 0.00}
        with pytest.raises(ValidationError, match="speed"):
            PetFeederEvent.model_validate(data)

    def test_invalid_posture_enum(self, valid_eating_output):
        """Invalid posture enum value should fail."""
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["body_signals"] = {"posture": "sleeping", "ear_position": "forward"}
        with pytest.raises(ValidationError):
            PetFeederEvent.model_validate(data)

    def test_invalid_species(self, valid_eating_output):
        """Invalid species should fail."""
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"], "species": "hamster"}
        with pytest.raises(ValidationError):
            PetFeederEvent.model_validate(data)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pet_schema.models'` or `ImportError`

- [ ] **Step 4: Implement models.py**

```python
"""Pydantic v2 models for PetFeederEvent schema v1.0."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class ActionDistribution(BaseModel):
    """Action probability distribution. All 6 values must sum to 1.0 ± 0.01."""

    eating: float
    drinking: float
    sniffing_only: float
    leaving_bowl: float
    sitting_idle: float
    other: float

    @model_validator(mode="after")
    def check_sum(self) -> ActionDistribution:
        """Validate that distribution values sum to 1.0 ± 0.01."""
        total = (
            self.eating + self.drinking + self.sniffing_only
            + self.leaving_bowl + self.sitting_idle + self.other
        )
        if abs(total - 1.0) > 0.01:
            msg = f"action distribution sum is {total:.4f}, must be 1.0 ± 0.01"
            raise ValueError(msg)
        return self


ActionLabel = Literal[
    "eating", "drinking", "sniffing_only", "leaving_bowl", "sitting_idle", "other"
]


class ActionInfo(BaseModel):
    """Pet action with primary label and probability distribution."""

    primary: ActionLabel
    distribution: ActionDistribution


class SpeedDistribution(BaseModel):
    """Eating speed distribution. All 3 values must sum to 1.0 ± 0.01."""

    fast: float
    normal: float
    slow: float

    @model_validator(mode="after")
    def check_sum(self) -> SpeedDistribution:
        """Validate that speed values sum to 1.0 ± 0.01."""
        total = self.fast + self.normal + self.slow
        if abs(total - 1.0) > 0.01:
            msg = f"speed distribution sum is {total:.4f}, must be 1.0 ± 0.01"
            raise ValueError(msg)
        return self


class EatingMetrics(BaseModel):
    """Eating behavior metrics."""

    speed: SpeedDistribution
    engagement: float
    abandoned_midway: float


class Mood(BaseModel):
    """Pet mood indicators. Each is independent 0-1 score."""

    alertness: float
    anxiety: float
    engagement: float


PostureLabel = Literal["relaxed", "tense", "hunched", "unobservable"]
EarPositionLabel = Literal["forward", "flat", "rotating", "unobservable"]


class BodySignals(BaseModel):
    """Observable body signals — enum values, not distributions."""

    posture: PostureLabel
    ear_position: EarPositionLabel


class AnomalySignals(BaseModel):
    """Anomaly detection signals. Each is independent 0-1 score."""

    vomit_gesture: float
    food_rejection: float
    excessive_sniffing: float
    lethargy: float
    aggression: float


class PetInfo(BaseModel):
    """Complete pet information for a single detected pet."""

    species: Literal["cat", "dog", "unknown"]
    breed_estimate: str
    id_tag: str
    id_confidence: float
    action: ActionInfo
    eating_metrics: EatingMetrics
    mood: Mood
    body_signals: BodySignals
    anomaly_signals: AnomalySignals


class BowlInfo(BaseModel):
    """Bowl state information."""

    food_fill_ratio: float | None = None
    water_fill_ratio: float | None = None
    food_type_visible: Literal["dry", "wet", "mixed", "unknown"]


class SceneInfo(BaseModel):
    """Scene metadata."""

    lighting: Literal["bright", "dim", "infrared_night"]
    image_quality: Literal["clear", "blurry", "partially_occluded"]
    confidence_overall: float


class PetFeederEvent(BaseModel):
    """Top-level VLM output schema for a single frame analysis."""

    schema_version: Literal["1.0"]
    pet_present: bool
    pet_count: int
    pet: PetInfo | None = None
    bowl: BowlInfo
    scene: SceneInfo
    narrative: str
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: ALL PASS

- [ ] **Step 6: Run lint**

```bash
make lint
```

Expected: ruff and mypy pass

- [ ] **Step 7: Commit**

```bash
git add src/pet_schema/models.py tests/conftest.py tests/test_models.py
git commit -m "feat(pet-schema): add Pydantic v2 models with distribution sum validators"
```

---

### Task 4: Validator — validator.py (TDD)

**Files:**
- Create: `src/pet_schema/validator.py`
- Create: `tests/test_validator.py`

- [ ] **Step 1: Write failing tests for validator**

```python
"""Tests for VLM output validator."""
import json

import pytest

from pet_schema.validator import ValidationResult, validate_output


class TestValidateOutputValid:
    """Test valid outputs pass validation."""

    def test_valid_eating_event(self, valid_eating_output):
        result = validate_output(json.dumps(valid_eating_output))
        assert result.valid is True
        assert result.errors == []

    def test_valid_no_pet_event(self, valid_no_pet_output):
        result = validate_output(json.dumps(valid_no_pet_output))
        assert result.valid is True
        assert result.errors == []


class TestValidateOutputInvalidJson:
    """Test invalid JSON handling."""

    def test_invalid_json_string(self):
        result = validate_output("not json at all")
        assert result.valid is False
        assert any("JSON" in e for e in result.errors)

    def test_empty_string(self):
        result = validate_output("")
        assert result.valid is False


class TestValidateOutputSchemaErrors:
    """Test JSON Schema level validation errors."""

    def test_missing_required_field(self, valid_eating_output):
        data = valid_eating_output.copy()
        del data["narrative"]
        result = validate_output(json.dumps(data))
        assert result.valid is False

    def test_invalid_enum_value(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["body_signals"] = {"posture": "sleeping", "ear_position": "forward"}
        result = validate_output(json.dumps(data))
        assert result.valid is False


class TestValidateOutputExtraValidations:
    """Test code-level extra validations."""

    def test_pet_present_true_but_pet_null(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["pet"] = None
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("pet_present" in e for e in result.errors)

    def test_pet_present_false_but_pet_not_null(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["pet_present"] = False
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("pet_present" in e for e in result.errors)

    def test_distribution_sum_too_high(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["action"] = {**data["pet"]["action"]}
        data["pet"]["action"]["distribution"] = {
            "eating": 0.80, "drinking": 0.10,
            "sniffing_only": 0.10, "leaving_bowl": 0.10,
            "sitting_idle": 0.00, "other": 0.00,
        }
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("distribution" in e for e in result.errors)

    def test_speed_sum_too_high(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["eating_metrics"] = {**data["pet"]["eating_metrics"]}
        data["pet"]["eating_metrics"]["speed"] = {"fast": 0.50, "normal": 0.50, "slow": 0.50}
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("speed" in e for e in result.errors)

    def test_primary_not_max_in_distribution(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["pet"] = {**data["pet"]}
        data["pet"]["action"] = {
            "primary": "drinking",
            "distribution": {
                "eating": 0.76, "drinking": 0.00,
                "sniffing_only": 0.14, "leaving_bowl": 0.05,
                "sitting_idle": 0.03, "other": 0.02,
            },
        }
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("primary" in e for e in result.errors)

    def test_narrative_too_long(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["narrative"] = "这" * 81
        result = validate_output(json.dumps(data))
        assert result.valid is False
        assert any("narrative" in e for e in result.errors)

    def test_narrative_exactly_80(self, valid_eating_output):
        data = valid_eating_output.copy()
        data["narrative"] = "这" * 80
        result = validate_output(json.dumps(data))
        assert result.valid is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_validator.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement validator.py**

```python
"""VLM output validator — JSON Schema + extra business constraints."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

VERSIONS_DIR = Path(__file__).parent.parent.parent / "versions"


@dataclass
class ValidationResult:
    """Result of validating a VLM output string."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_output(json_str: str, version: str = "1.0") -> ValidationResult:
    """Validate a VLM JSON output string.

    Runs JSON Schema validation followed by extra business-rule checks.

    Args:
        json_str: Raw JSON string from VLM output.
        version: Schema version to validate against.

    Returns:
        ValidationResult with valid=True if all checks pass.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ValidationResult(valid=False, errors=[f"JSON 解析失败: {e}"])

    schema_path = VERSIONS_DIR / version / "schema.json"
    if not schema_path.exists():
        return ValidationResult(
            valid=False, errors=[f"Schema 版本 {version} 不存在: {schema_path}"]
        )
    schema = json.loads(schema_path.read_text())

    errors: list[str] = []
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema 验证失败: {e.message}")

    errors.extend(_extra_validations(data))

    warnings: list[str] = []
    scene = data.get("scene", {})
    if isinstance(scene.get("confidence_overall"), float) and scene["confidence_overall"] < 0.5:
        warnings.append(f"confidence_overall 偏低: {scene['confidence_overall']}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _extra_validations(data: dict) -> list[str]:
    """Business-rule validations that JSON Schema cannot express.

    Checks:
        - pet_present / pet field consistency
        - action.distribution sum = 1.0 ± 0.01
        - eating_metrics.speed sum = 1.0 ± 0.01
        - action.primary matches highest probability in distribution
        - narrative length <= 80 chars

    Args:
        data: Parsed JSON dict.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []

    if data.get("pet_present") and data.get("pet") is None:
        errors.append("pet_present=true 但 pet 字段为 null")
    if not data.get("pet_present") and data.get("pet") is not None:
        errors.append("pet_present=false 但 pet 字段非 null")

    pet = data.get("pet")
    if pet:
        dist = pet.get("action", {}).get("distribution", {})
        if dist:
            total = sum(float(v) for v in dist.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"action.distribution 求和 {total:.4f} 超出 1.0±0.01"
                )

        primary = pet.get("action", {}).get("primary")
        if dist and primary:
            max_key = max(dist, key=lambda k: float(dist[k]))
            if primary != max_key:
                errors.append(
                    f"action.primary '{primary}' 不是 distribution 中最高概率的 key '{max_key}'"
                )

        speed = pet.get("eating_metrics", {}).get("speed", {})
        if speed:
            total = sum(float(v) for v in speed.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"eating_metrics.speed 求和 {total:.4f} 超出 1.0±0.01"
                )

    narrative = data.get("narrative", "")
    if len(narrative) > 80:
        errors.append(f"narrative 长度 {len(narrative)} 字超过 80 字限制")

    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validator.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run lint**

```bash
make lint
```

- [ ] **Step 6: Commit**

```bash
git add src/pet_schema/validator.py tests/test_validator.py
git commit -m "feat(pet-schema): add validator with jsonschema + extra business constraints"
```

---

### Task 5: Renderer — renderer.py (TDD)

**Files:**
- Create: `src/pet_schema/renderer.py`
- Create: `tests/test_renderer.py`

- [ ] **Step 1: Write failing tests for renderer**

```python
"""Tests for prompt renderer."""
import pytest

from pet_schema.renderer import render_prompt


class TestRenderPrompt:
    """Test prompt rendering."""

    def test_returns_system_and_user(self):
        system, user = render_prompt()
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_system_prompt_contains_key_instructions(self):
        system, _ = render_prompt()
        assert "JSON" in system
        assert "不拟人化" in system
        assert "80" in system

    def test_user_prompt_with_few_shot(self):
        _, user = render_prompt(few_shot=True)
        assert "参考示例" in user
        assert "british_shorthair" in user

    def test_user_prompt_without_few_shot(self):
        _, user = render_prompt(few_shot=False)
        assert "参考示例" not in user

    def test_invalid_version_raises(self):
        with pytest.raises(FileNotFoundError):
            render_prompt(version="99.0")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_renderer.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement renderer.py**

```python
"""Prompt renderer — loads and renders system/user prompts from versioned templates."""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

VERSIONS_DIR = Path(__file__).parent.parent.parent / "versions"


def render_prompt(
    version: str = "1.0", few_shot: bool = True
) -> tuple[str, str]:
    """Render system and user prompts for a given schema version.

    Args:
        version: Schema version directory to load from.
        few_shot: Whether to inject few-shot examples into the user prompt.

    Returns:
        Tuple of (system_prompt, user_prompt).

    Raises:
        FileNotFoundError: If the version directory or required files don't exist.
    """
    version_dir = VERSIONS_DIR / version
    system_path = version_dir / "prompt_system.txt"
    user_template_path = version_dir / "prompt_user.jinja2"
    examples_path = version_dir / "few_shot_examples.json"

    if not version_dir.exists():
        msg = f"Schema 版本目录不存在: {version_dir}"
        raise FileNotFoundError(msg)
    if not system_path.exists():
        msg = f"System prompt 不存在: {system_path}"
        raise FileNotFoundError(msg)
    if not user_template_path.exists():
        msg = f"User prompt 模板不存在: {user_template_path}"
        raise FileNotFoundError(msg)

    system_prompt = system_path.read_text(encoding="utf-8")

    template = Template(user_template_path.read_text(encoding="utf-8"))

    few_shot_examples = ""
    if few_shot and examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
        few_shot_examples = json.dumps(examples, ensure_ascii=False, indent=2)

    user_prompt = template.render(few_shot_examples=few_shot_examples if few_shot else "")

    return system_prompt, user_prompt
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_renderer.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Run lint**

```bash
make lint
```

- [ ] **Step 6: Commit**

```bash
git add src/pet_schema/renderer.py tests/test_renderer.py
git commit -m "feat(pet-schema): add prompt renderer with Jinja2 template support"
```

---

### Task 6: Public API — __init__.py

**Files:**
- Modify: `src/pet_schema/__init__.py`

- [ ] **Step 1: Update __init__.py to expose public API**

```python
"""pet-schema: VLM output schema definition, validation, and prompt rendering.

Public API:
    validate_output — Validate a VLM JSON output string against the schema.
    render_prompt — Render system and user prompts for a given schema version.
"""
from pet_schema.renderer import render_prompt
from pet_schema.validator import validate_output

__all__ = ["validate_output", "render_prompt"]
```

- [ ] **Step 2: Verify imports work**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
python -c "from pet_schema import validate_output, render_prompt; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pet_schema/__init__.py
git commit -m "feat(pet-schema): expose validate_output and render_prompt as public API"
```

---

### Task 7: test_examples.py — few-shot examples self-validation

**Files:**
- Create: `tests/test_examples.py`

- [ ] **Step 1: Write test_examples.py**

```python
"""Tests that all few-shot examples pass schema validation and Pydantic parsing."""
import json

from pet_schema import validate_output
from pet_schema.models import PetFeederEvent


class TestFewShotExamples:
    """Every few-shot example must be valid under both validation paths."""

    def test_all_examples_pass_validator(self, v1_examples):
        for i, example in enumerate(v1_examples):
            output_str = json.dumps(example["output"])
            result = validate_output(output_str)
            assert result.valid, (
                f"Example {i} ({example['scene_desc']}) failed validation: {result.errors}"
            )

    def test_all_examples_pass_pydantic(self, v1_examples):
        for i, example in enumerate(v1_examples):
            event = PetFeederEvent.model_validate(example["output"])
            assert event.schema_version == "1.0", f"Example {i} has wrong schema_version"

    def test_no_pet_example_has_null_pet(self, v1_examples):
        for example in v1_examples:
            output = example["output"]
            if not output["pet_present"]:
                assert output["pet"] is None, (
                    f"Example '{example['scene_desc']}': pet_present=false but pet is not null"
                )

    def test_pet_example_has_pet(self, v1_examples):
        for example in v1_examples:
            output = example["output"]
            if output["pet_present"]:
                assert output["pet"] is not None, (
                    f"Example '{example['scene_desc']}': pet_present=true but pet is null"
                )
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_examples.py -v
```

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_examples.py
git commit -m "test(pet-schema): add few-shot examples self-validation tests"
```

---

### Task 8: test_pydantic_sync.py — four-direction anti-drift tests

**Files:**
- Create: `tests/test_pydantic_sync.py`

- [ ] **Step 1: Write four-direction sync tests**

```python
"""Four-direction anti-drift tests between schema.json and Pydantic models.

Direction 1: Pydantic → JSON Schema field sets must match
Direction 2: JSON Schema examples → Pydantic parsing must succeed
Direction 3: Pydantic model_dump → jsonschema.validate must pass
Direction 4: Enum values in schema.json == Literal values in Pydantic
"""
import json

import jsonschema

from pet_schema.models import (
    ActionDistribution,
    ActionLabel,
    BodySignals,
    BowlInfo,
    EarPositionLabel,
    PetFeederEvent,
    PostureLabel,
    SceneInfo,
)


class TestDirection1PydanticToSchemaFields:
    """Pydantic model_json_schema() required/properties must match schema.json."""

    def test_top_level_required_fields(self, v1_schema):
        pydantic_schema = PetFeederEvent.model_json_schema()
        schema_required = set(v1_schema["required"])
        pydantic_required = set(pydantic_schema.get("required", []))
        assert schema_required == pydantic_required, (
            f"Required mismatch: schema={schema_required}, pydantic={pydantic_required}"
        )

    def test_top_level_property_names(self, v1_schema):
        pydantic_schema = PetFeederEvent.model_json_schema()
        schema_props = set(v1_schema["properties"].keys())
        pydantic_props = set(pydantic_schema.get("properties", {}).keys())
        assert schema_props == pydantic_props, (
            f"Property mismatch: schema={schema_props}, pydantic={pydantic_props}"
        )


class TestDirection2SchemaExamplesToPydantic:
    """few_shot_examples data must parse through Pydantic."""

    def test_all_examples_parse(self, v1_examples):
        for i, example in enumerate(v1_examples):
            event = PetFeederEvent.model_validate(example["output"])
            assert event.schema_version == "1.0", f"Example {i} parse failed"


class TestDirection3PydanticDumpToJsonschema:
    """Pydantic model_dump() output must pass jsonschema.validate()."""

    def test_valid_event_roundtrip(self, v1_schema, valid_eating_output):
        event = PetFeederEvent.model_validate(valid_eating_output)
        dumped = event.model_dump(mode="python")
        jsonschema.validate(dumped, v1_schema)

    def test_no_pet_roundtrip(self, v1_schema, valid_no_pet_output):
        event = PetFeederEvent.model_validate(valid_no_pet_output)
        dumped = event.model_dump(mode="python")
        jsonschema.validate(dumped, v1_schema)


class TestDirection4EnumSync:
    """Enum arrays in schema.json must equal Literal values in Pydantic."""

    def _get_pet_schema(self, v1_schema) -> dict:
        """Extract the pet object schema from oneOf."""
        for option in v1_schema["properties"]["pet"]["oneOf"]:
            if option.get("type") == "object":
                return option
        raise AssertionError("No object type in pet oneOf")

    def test_species_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(pet_schema["properties"]["species"]["enum"])
        pydantic_enum = {"cat", "dog", "unknown"}
        assert schema_enum == pydantic_enum

    def test_action_primary_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(pet_schema["properties"]["action"]["properties"]["primary"]["enum"])
        pydantic_enum = set(ActionLabel.__args__)
        assert schema_enum == pydantic_enum

    def test_action_distribution_keys(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_keys = set(
            pet_schema["properties"]["action"]["properties"]["distribution"]["required"]
        )
        pydantic_keys = set(ActionDistribution.model_fields.keys())
        assert schema_keys == pydantic_keys

    def test_posture_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(
            pet_schema["properties"]["body_signals"]["properties"]["posture"]["enum"]
        )
        pydantic_enum = set(PostureLabel.__args__)
        assert schema_enum == pydantic_enum

    def test_ear_position_enum(self, v1_schema):
        pet_schema = self._get_pet_schema(v1_schema)
        schema_enum = set(
            pet_schema["properties"]["body_signals"]["properties"]["ear_position"]["enum"]
        )
        pydantic_enum = set(EarPositionLabel.__args__)
        assert schema_enum == pydantic_enum

    def test_food_type_enum(self, v1_schema):
        schema_enum = set(
            v1_schema["properties"]["bowl"]["properties"]["food_type_visible"]["enum"]
        )
        pydantic_fields = BowlInfo.model_fields
        # Extract Literal args from food_type_visible annotation
        annotation = pydantic_fields["food_type_visible"].annotation
        pydantic_enum = set(annotation.__args__)
        assert schema_enum == pydantic_enum

    def test_lighting_enum(self, v1_schema):
        schema_enum = set(
            v1_schema["properties"]["scene"]["properties"]["lighting"]["enum"]
        )
        annotation = SceneInfo.model_fields["lighting"].annotation
        pydantic_enum = set(annotation.__args__)
        assert schema_enum == pydantic_enum

    def test_image_quality_enum(self, v1_schema):
        schema_enum = set(
            v1_schema["properties"]["scene"]["properties"]["image_quality"]["enum"]
        )
        annotation = SceneInfo.model_fields["image_quality"].annotation
        pydantic_enum = set(annotation.__args__)
        assert schema_enum == pydantic_enum
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_pydantic_sync.py -v
```

Expected: ALL PASS (if schema.json and models.py are in sync). If any test fails, fix the mismatch in either schema.json or models.py, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pydantic_sync.py
git commit -m "test(pet-schema): add four-direction anti-drift sync tests"
```

---

### Task 9: Full test suite and lint pass

**Files:**
- Modify: `tests/test_models.py` (cleanup — remove if fully covered by other tests, or keep as unit tests)

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
make test
```

Expected: ALL PASS

- [ ] **Step 2: Run full lint**

```bash
make lint
```

Expected: ruff and mypy both pass with zero errors

- [ ] **Step 3: Fix any issues found, re-run until clean**

- [ ] **Step 4: Clean up test_models.py**

If test_models.py tests are fully covered by test_validator.py / test_examples.py / test_pydantic_sync.py, remove duplicates. Keep any unique model-level unit tests (e.g. distribution sum validation error messages).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore(pet-schema): full test suite and lint pass"
```

---

### Task 10: Generate requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Generate locked requirements**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-schema
pip-compile pyproject.toml -o requirements.txt --strip-extras
```

- [ ] **Step 2: Verify install from requirements.txt**

```bash
pip install -r requirements.txt
make test
```

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore(pet-schema): add pip-compiled requirements.txt"
```
