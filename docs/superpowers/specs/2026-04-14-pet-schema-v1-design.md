# pet-schema v1.0 设计文档

## 概述

pet-schema 是 Train-Pet-Pipeline 全链的上游合同仓库，定义 VLM 输出的 JSON Schema、Prompt 模板、验证器和 Pydantic 模型。所有下游仓库（pet-data、pet-annotation、pet-train、pet-eval、pet-quantize、pet-ota）依赖本仓库。

本文档基于 `pet-infra/docs/DEVELOPMENT_GUIDE.md` 中 pet-schema 的规范，记录经过讨论后的设计决策和调整。

## 设计决策

### 决策 1：部分字段从概率分布降级为枚举

**背景**：文档原版要求 4 组概率分布都精确求和为 1.0。2B 端侧模型同时维持多组分布约束训练难度高，且部分字段在固定俯角摄像头下信息量低。

**调整**：

| 字段 | 原设计 | 新设计 | 理由 |
|---|---|---|---|
| `body_signals.posture` | 3 值概率分布（求和=1） | 枚举 `relaxed\|tense\|hunched\|unobservable` | 俯视角度下 relaxed/tense 区分度低 |
| `body_signals.ear_position` | 4 值概率分布（求和=1） | 枚举 `forward\|flat\|rotating\|unobservable` | 进食低头时耳朵几乎不可见 |

**保留为概率分布的字段**（高信息量）：
- `action.distribution`（6 值）-- 核心行为分类
- `eating_metrics.speed`（3 值）-- 进食速度有明确临床意义

### 决策 2：删除 mood.comfort

**背景**：`comfort` 与 `anxiety` 高度负相关，同时要求 2B 模型输出两个几乎互为反数的值，增加训练难度却信息冗余。

**调整**：mood 从 4 维降为 3 维：`alertness`、`anxiety`、`engagement`。各自独立 0-1 评分，不做求和约束。

### 决策 3：narrative 字数限制从 50 放宽到 80

**背景**：narrative 是逐帧描述（非逐事件），由 2B 端侧模型生成。用户最终看到的是设备端/云端聚合后的事件描述，不是单帧 narrative。保留字数限制有助于约束 2B 模型输出质量，但 50 字在异常场景下偏紧。

**调整**：`maxLength` 从 50 改为 80。

### 决策 4：id_tag 多宠识别方案保持现有设计

**背景**：基于外观特征描述（如 `orange_tabby_large`）的方案在同色多宠、幼猫成长、夜视模式下有局限。但 v1.0 阶段众筹早期以单宠用户为主，`id_confidence` 低时可以单独处理。

**调整**：不调整，v1.0 接受局限。

### 决策 5：不新增 drinking_metrics

**背景**：饮水行为变化是肾病/糖尿病的早期信号，行为学上有意义。但与 eating_metrics 结构重复，且 v1.0 的核心目标是进食监控。

**调整**：不新增。后续版本按需加入。

### 决策 6：方案 C -- 双向校验的混合方案

**背景**：schema.json 和 models.py 是两份定义，需要保证一致性。

- schema.json 是权威来源（对外合同，下游仓库和 CI 直接消费）
- models.py 是 Pydantic v2 内部使用模型
- test_pydantic_sync.py 做四方向双向校验，CI 阻断不一致的合并

## 项目结构

```
pet-schema/
├── versions/
│   └── v1.0/
│       ├── prompt_system.txt          # system prompt 纯文本
│       ├── prompt_user.jinja2         # user prompt Jinja2 模板
│       ├── schema.json                # JSON Schema Draft-7（权威来源）
│       ├── few_shot_examples.json     # 3 个示例
│       └── CHANGELOG.md
├── src/
│   └── pet_schema/
│       ├── __init__.py                # 暴露 validate_output, render_prompt
│       ├── validator.py               # jsonschema + 额外约束验证
│       ├── renderer.py                # Jinja2 渲染 prompt
│       └── models.py                  # Pydantic v2 模型
├── tests/
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

## Schema v1.0 字段定义

### 完整字段结构

```
PetFeederEvent
├── schema_version: "1.0"
├── pet_present: bool
├── pet_count: int (0-4)
├── pet: PetInfo | null
│   ├── species: "cat" | "dog" | "unknown"
│   ├── breed_estimate: str
│   ├── id_tag: str
│   ├── id_confidence: float (0.00-1.00)
│   ├── action: ActionInfo
│   │   ├── primary: ActionLabel                    # distribution 中最高概率的 key
│   │   └── distribution: ActionDistribution        # 6 值，求和 1.0±0.01
│   │       eating / drinking / sniffing_only / leaving_bowl / sitting_idle / other
│   ├── eating_metrics: EatingMetrics
│   │   ├── speed: SpeedDistribution                # 3 值，求和 1.0±0.01
│   │   │   fast / normal / slow
│   │   ├── engagement: float (0.00-1.00)
│   │   └── abandoned_midway: float (0.00-1.00)
│   ├── mood: Mood                                  # 各自独立 0-1，不求和
│   │   ├── alertness: float (0.00-1.00)
│   │   ├── anxiety: float (0.00-1.00)
│   │   └── engagement: float (0.00-1.00)
│   ├── body_signals: BodySignals
│   │   ├── posture: "relaxed" | "tense" | "hunched" | "unobservable"       # 枚举
│   │   └── ear_position: "forward" | "flat" | "rotating" | "unobservable"  # 枚举
│   └── anomaly_signals: AnomalySignals             # 各自独立 0-1
│       ├── vomit_gesture: float (0.00-1.00)
│       ├── food_rejection: float (0.00-1.00)
│       ├── excessive_sniffing: float (0.00-1.00)
│       ├── lethargy: float (0.00-1.00)
│       └── aggression: float (0.00-1.00)
├── bowl: BowlInfo
│   ├── food_fill_ratio: float (0.00-1.00) | null
│   ├── water_fill_ratio: float (0.00-1.00) | null
│   └── food_type_visible: "dry" | "wet" | "mixed" | "unknown"
├── scene: SceneInfo
│   ├── lighting: "bright" | "dim" | "infrared_night"
│   ├── image_quality: "clear" | "blurry" | "partially_occluded"
│   └── confidence_overall: float (0.00-1.00)
└── narrative: str (maxLength: 80)
```

### 求和约束（代码层强制校验）

| 分布组 | 值数量 | 约束 |
|---|---|---|
| `action.distribution` | 6 | 求和 = 1.0 ± 0.01 |
| `eating_metrics.speed` | 3 | 求和 = 1.0 ± 0.01 |

### 其他代码层约束

- `pet_present=true` 时 `pet` 不能为 null；`pet_present=false` 时 `pet` 必须为 null
- `action.primary` 必须等于 `distribution` 中概率最高的 key
- `narrative` 长度 ≤ 80 字
- 所有 float 概率字段值域 [0.00, 1.00]，保留两位小数

## 模块设计

### validator.py

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

def validate_output(json_str: str, version: str = "1.0") -> ValidationResult:
    """
    验证 VLM 输出字符串。
    1. JSON 解析
    2. jsonschema 结构验证
    3. 额外业务约束验证
    返回 ValidationResult，errors 非空则 valid=False。
    """
```

额外验证 `_extra_validations(data)` 覆盖：
- pet_present 与 pet 字段的一致性
- action.distribution 求和
- eating_metrics.speed 求和
- action.primary 与 distribution 最高概率一致
- narrative 字数

### renderer.py

```python
def render_prompt(version: str = "1.0", few_shot: bool = True) -> tuple[str, str]:
    """
    渲染 prompt 模板。
    返回 (system_prompt, user_prompt)。
    few_shot=True 时 user_prompt 注入 few_shot_examples。
    版本不存在时抛出 FileNotFoundError。
    """
```

### models.py

Pydantic v2 模型，使用 `model_validator` 实现求和约束。模型层级：

- `PetFeederEvent` -- 顶层
- `PetInfo` -- 宠物信息
- `ActionInfo` / `ActionDistribution` -- 行为分布
- `EatingMetrics` / `SpeedDistribution` -- 进食指标
- `Mood` -- 情绪（3 维）
- `BodySignals` -- 体态信号（枚举）
- `AnomalySignals` -- 异常信号
- `BowlInfo` -- 碗状态
- `SceneInfo` -- 场景

### __init__.py

公开 API 仅两个函数：

```python
from pet_schema.validator import validate_output
from pet_schema.renderer import render_prompt
```

## 测试策略

### test_validator.py

- 3 种合法输出通过验证（正常进食、无宠物、异常场景）
- distribution 求和超出 ±0.01 → errors
- pet_present/pet 不一致 → errors
- primary 与 distribution 最高概率不一致 → errors
- narrative 超 80 字 → errors
- 非法 JSON → errors
- 非法枚举值 → errors

### test_renderer.py

- 返回非空 (system, user) 字符串
- system prompt 包含关键指令词
- few_shot=True 包含示例
- few_shot=False 不包含示例
- 不存在的版本号 → FileNotFoundError

### test_examples.py

- 每个 few_shot_example 通过 validate_output()
- 每个 few_shot_example 通过 PetFeederEvent.model_validate()
- pet_present=false 的条目 pet 为 null

### test_pydantic_sync.py（防漂移核心）

四方向校验：

1. **Pydantic → JSON Schema 字段集**：model_json_schema() 导出的 required/properties 与 schema.json 一致
2. **JSON Schema → Pydantic 解析**：few_shot_examples 数据能被 Pydantic 模型解析
3. **Pydantic → JSON Schema 验证**：model_dump() 输出通过 jsonschema.validate()
4. **枚举值同步**：schema.json 的 enum 数组与 Pydantic 的 Literal 类型值集合相等

## pyproject.toml

```toml
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

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

## 与文档原版的差异汇总

| 项目 | DEVELOPMENT_GUIDE 原版 | 本设计 | 理由 |
|---|---|---|---|
| body_signals.posture | 3 值概率分布 | 枚举 + unobservable | 俯角信息量低，减轻 2B 训练负担 |
| body_signals.ear_position | 4 值概率分布 | 枚举 + unobservable | 进食时几乎不可见 |
| mood.comfort | 存在 (0-1) | 删除 | 与 anxiety 负相关冗余 |
| narrative maxLength | 50 | 80 | 异常场景需要更多描述空间 |
| Pydantic 同步检查 | 单向一致性测试 | 四方向双向校验 | 更可靠地防止漂移 |

## 版本规则

遵循 DEVELOPMENT_GUIDE 的版本管理规则：
- 新增可选字段 → 小版本 v1.0 → v1.1
- 新增必填字段 → 大版本 v1.0 → v2.0
- 已有版本目录永不修改
