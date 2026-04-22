# pet-schema 技术设计文档

> 维护说明：代码变更影响本文档任一章节时，与代码在同 PR 内更新（参见 `feedback_devguide_sync` 原则）
> 最后对齐：pet-schema v2.4.0 / 2026-04-23

---

## 1. 仓库职责

pet-schema 是整个 Train-Pet-Pipeline 生态的**契约根节点**（chain head）。其唯一职责是定义所有跨仓库共享的数据类型、验证规则和迁移文件，下游 8 个仓库通过 `import pet_schema` 消费这些类型。

**在 pet-schema 做的：**

- Pydantic v2 数据模型（Sample / Annotation / ModelCard / ExperimentRecipe / Metric / Config）
- Alembic 数据库迁移文件（历史文件不可修改，只追加新文件）
- HuggingFace `datasets.Features` 适配器（`adapters/hf_features.py`）
- `SCHEMA_VERSION` 常量（唯一权威，parity-checked against `pyproject.toml`）
- VLM JSON 输出的运行时语义校验（`validator.py`）

**不在 pet-schema 做的：**

- 业务逻辑（训练循环、推理、标注路由）——这些在各下游仓库
- I/O 操作（文件读写、数据库直连）——下游通过 `store.py` 操作
- 插件注册表 / CLI——住在 pet-infra 或各功能仓库
- pet-id 的 PetCard 注册机制（pet-id 有自己的 registry，不依赖 pet-schema）

作为链头，pet-schema 本身**没有任何 `pet-*` 上游依赖**。

---

## 2. 输入输出契约

### 上游

无（chain root）。

### 下游消费方

所有下游通过安装 `pet-schema` 包并 `import pet_schema` 消费类型。变更 pet-schema 时，`schema_guard.yml` 会向下游 7 个仓库派发 `schema-updated` repository_dispatch，触发它们的 CI 校验。

| 下游仓库 | 消费的主要类型 |
|---|---|
| pet-infra | `ExperimentRecipe`, `ModelCard`, `ResourceSpec`, `RecipeStage`, `ArtifactRef`, `TrainerConfig`, `EvaluatorConfig`, `ConverterConfig` |
| pet-data | `Sample`, `VisionSample`, `AudioSample`, `SensorSample`, `PetFeederEvent`（legacy v1）, `Annotation` 变体 |
| pet-annotation | `BaseAnnotation` + 4 discriminator 变体（`LLMAnnotation` / `ClassifierAnnotation` / `RuleAnnotation` / `HumanAnnotation`）, `DpoPair` |
| pet-train | `ExperimentRecipe`, `ModelCard`, `TrainerConfig` |
| pet-eval | `ModelCard`, `GateCheck`, `MetricResult`, `EvaluationReport`, `PetFeederEvent`（legacy） |
| pet-quantize | `EdgeArtifact`, `DeploymentStatus`, `ModelCard`, `QuantConfig` |
| pet-ota | `ModelCard`, `EdgeArtifact`, `DeploymentStatus`，以及 `ModelCard.to_manifest_entry()` 用于 manifest.json |
| pet-id | 不依赖 pet-schema（使用自己的 `PetCard` registry） |

### 契约变更流程

```
PR 提到 pet-schema
  → CI (schema-validation.yml) 本地验证
  → merge to main
  → schema_guard.yml push-trigger
  → repository_dispatch → 7 个下游仓库各自跑 schema_validate CI
```

pet-schema 的 PR 需要至少两位 reviewer approve（比普通仓库多一位），原因是任何字段名 / 类型变更都是跨仓库破坏性事件。

---

## 3. 架构总览

```
src/pet_schema/
├── __init__.py              — 公开 API re-exports（所有下游通过此处导入）
├── version.py               — SCHEMA_VERSION 常量（与 pyproject.toml parity-checked）
├── enums.py                 — 共享枚举（Modality, PetSpecies, BowlType, …）
├── samples.py               — BaseSample + VisionSample / AudioSample / SensorSample，
│                              Sample = Annotated[…, Discriminator("modality")]
├── models.py                — PetFeederEvent（legacy v1，保留以支持 pet-eval 现有路径）
├── annotations.py           — BaseAnnotation + 4 discriminator 变体，DpoPair
├── model_card.py            — ModelCard, ResourceSpec, QuantConfig, EdgeArtifact,
│                              HardwareValidation, DeploymentStatus
├── recipe.py                — ExperimentRecipe, RecipeStage, ArtifactRef, AblationAxis
│                              + to_dag()（nx.DiGraph DAG 构造与环检测）
├── configs.py               — Hydra 结构化配置外壳（TrainerConfig, EvaluatorConfig,
│                              ConverterConfig, DatasetConfig, ResourcesSection）
├── metric.py                — MetricResult, GateCheck, EvaluationReport
│                              + GateCheck.evaluate() 内置 abs_tol=1e-6
├── validator.py             — validate_output()：VLM JSON 语义校验 + 业务规则检查
├── renderer.py              — render_prompt()：Jinja2 模板渲染标准 prompt
├── adapters/
│   └── hf_features.py       — Pydantic model → HuggingFace datasets.Features 转换
└── versions/
    └── v1.0/
        └── schema.json      — VLM 输出 JSON Schema（versioned）
```

```
tests/                       — pytest suite（167 个测试，0 lint errors）
.github/workflows/
├── ci.yml                   — ruff lint + mypy + pytest
└── schema_guard.yml         — merge to main 后派发到 7 个下游仓库
```

**关键数据流（正常路径）：**

```
下游仓库调用 model_validate(data)
  → Pydantic v2 intra-model 字段约束检查
  → （可选）validate_output(json_str) 跨模型语义检查
  → 合规数据对象
```

---

## 4. 核心模块详解

### 4.1 `annotations.py` — 4 范式 Annotation discriminator

**是什么：** `BaseAnnotation` 基类 + 4 个具体子类（`LLMAnnotation` / `ClassifierAnnotation` / `RuleAnnotation` / `HumanAnnotation`）+ `DpoPair`。`Annotation` 类型别名通过 `Discriminator("annotator_type")` 完成分发。

**为什么这样设计（Why）：**

Phase 2 之前，所有 annotation 共用单一模型，pet-annotation 的插件路由需要在运行时 `if annotator_type == "llm"` 做类型分支。4 范式设计让插件注册表可以直接按类型路由，无需 type switch，也让每种生产者的特有字段（`LLMAnnotation.prompt_hash`, `ClassifierAnnotation.class_probs`, …）有了类型安全保障。

**权衡（Tradeoff）：**

- 考虑过：每个生产者类型独立表（separate tables per producer），pet-annotation 端 4 张表完全独立。
- 为什么拒绝：矩阵评估（如对比同一 target 的 LLM 和 human annotation）需要跨类型查询；独立表会导致 join 地狱或需要维护 union view。
- 当前方案：单表多态（polymorphic single table），discriminator 在 Python 层分发。数据库层保留统一表结构，查询简单。

**新工程师易踩坑（Pitfall）：**

1. `BaseAnnotation` 设置了 `extra="forbid"`（2026-04-23 新增）。新增生产者类型必须：(a) 继承 `BaseAnnotation`，(b) 在 `annotator_type` 字段声明自己的 `Literal[...]`，(c) 加入 `Annotation` 联合类型，(d) 写测试覆盖 extra=forbid 拒绝行为。只做 (a) 不做 (b)(c) 会导致 discriminator 无法分发到新类型。
2. `DpoPair` 和 `BaseAnnotation` 都用了 `extra="forbid"`，但 `DpoPair` 不继承 `BaseAnnotation`——它是独立的契约（pair-level，不是 sample-level）。

---

### 4.2 `recipe.py` — RecipeStage 的 `inputs` + `depends_on` 双机制

**是什么：** `RecipeStage` 有两个容易混淆的字段：

- `inputs: dict[str, ArtifactRef]`：有类型的输入绑定，声明该 stage 消费哪些制品（`ArtifactRef` 有 `ref_type` 和 `ref_value`）。
- `depends_on: list[str]`：DAG 调度顺序，声明哪些 stage 必须先于本 stage 完成。

`ExperimentRecipe.to_dag()` 用 `depends_on` 构造 `nx.DiGraph`，用 networkx 检测环。

**为什么这样设计（Why）：**

两者目的不同：`inputs` 是类型绑定（orchestrator 知道"喂什么进来"），`depends_on` 是调度约束（orchestrator 知道"什么顺序运行"）。它们在语义上可能重叠（一个 `ref_type="recipe_stage_output"` 的 input 隐含依赖），但两者都是必须的，因为：

- 有时 stage 之间存在调度依赖但不存在直接数据输入（例如：stage B 必须等 stage A 写完 checkpoint 才能开始，但 B 不直接消费 A 的输出）。
- `to_dag()` 只读 `depends_on`——如果合并两者，orchestrator 就必须解析 `ArtifactRef` 内部结构来推断依赖，这将 orchestration 逻辑渗入 schema 验证。

**权衡（Tradeoff）：**

- 考虑过：从 `inputs` 推导 `depends_on`（`ref_type="recipe_stage_output"` 时自动提取 `ref_value` 作为依赖）。
- 为什么拒绝：这会让 `to_dag()` 依赖 `ArtifactRef.ref_type` 的语义——schema 层（数据契约）要了解 orchestration 层（执行语义），违反职责分离。另外调度依赖并不总能从 inputs 推断。
- 当前方案：冗余但明确，工程师手动维护两者一致性。

**新工程师易踩坑（Pitfall）：**

新建 stage 时如果某个 `inputs` 条目的 `ref_type` 是 `"recipe_stage_output"`，必须**同时**在 `depends_on` 列出那个 stage 名。只填 `inputs` 不填 `depends_on`，`to_dag()` 算出的依赖关系会不完整，`ExperimentRecipe` 的 `@model_validator` 会漏掉环。CI 环检测会失效，runtime 才暴露问题。

---

### 4.3 `model_card.py` — ModelCard 与 ResourceSpec 的分离

**是什么：** `ModelCard` 是模型的规范性描述卡，包含身份 / 可复现性 / 制品 / 评估指标 / 部署历史等字段。`ResourceSpec`（同文件）是一个独立模型，声明训练所需硬件资源（`gpu_count`, `gpu_memory_gb`, `cpu_count`, `estimated_hours`）。

**为什么这样设计（Why）：**

`ResourceSpec` 是 `pet_infra.base.trainer.TrainerBase.estimate_resources()` 的返回类型——它是 trainer 插件的**预飞契约**（pre-run declaration），在训练开始前声明期望资源。`ModelCard` 是**事后描述**（post-run record），记录实际训练产出的元数据。两者生命周期不同：ResourceSpec 在 run 之前由 trainer 插件计算，ModelCard 在 run 完成后写入。

注意：`ResourcesSection`（`configs.py`）是另一个概念——它是 `TrainerConfig` 的子字段，提供 Hydra 配置层的资源默认值和上限，不是运行时估算值。三者（`ResourceSpec` / `ResourcesSection` / `ModelCard.metrics`）三处资源相关概念容易混淆。

**权衡（Tradeoff）：**

- 考虑过：将 `ResourceSpec` 嵌入 `ModelCard`，作为 `actual_resources: ResourceSpec | None` 字段，记录实际使用量。
- 为什么拒绝：实际资源使用量由 ClearML 任务在 run 中动态采集，不属于 schema-level 静态记录。嵌入会让 ModelCard 混合"描述性"与"运行时测量"两类信息，破坏其描述性语义。
- 当前方案：实际使用量在 ClearML 任务中追踪，ModelCard 只保存 `clearml_task_id` 作为追踪指针。

**新工程师易踩坑（Pitfall）：**

新工程师看到 `ResourceSpec` 时常问"为什么不在 ModelCard 里？"——答案是：ResourceSpec 是 trainer plugin 的接口返回值（`estimate_resources() → ResourceSpec`），用于 orchestrator preflight 检查，和 ModelCard 完全不同的使用场景。如果要在代码里用 ResourceSpec，应该调用 `trainer.estimate_resources()`，而不是从 ModelCard 读取。

---

### 4.4 `metric.py` — GateCheck 的 `abs_tol=1e-6`

**是什么：** `GateCheck` 是生产质量门检测，`GateCheck.evaluate()` 比较 `actual` 与 `threshold`，支持 `ge` / `le` / `eq` 三种比较器。`eq` 使用 `math.isclose(actual, threshold, abs_tol=1e-6)` 而非直接 `==`。

**为什么这样设计（Why）：**

`==` 比较浮点数会产生误报：例如 `mean([0.85, 0.85]) == 0.85` 在 Python 中偶尔返回 `False`（浮点累积）。`abs_tol=1e-6` 覆盖正常的浮点算术误差（量级 ~1e-9 到 1e-15），同时对真实的指标回退（量级 ~1e-3 到 1e-2）仍然严格。

**权衡（Tradeoff）：**

- 考虑过：将 `abs_tol` 参数化，允许每个 GateCheck 实例自定义容差。
- 为什么拒绝：YAGNI——所有已观察到的浮点误差都在 1e-6 范围内处理。参数化增加调用方的配置负担，且容易被错误使用（设太大会掩盖真实回退）。
- 保留了 `rel_tol` 默认值 0（即 `math.isclose` 的 `rel_tol=0`），因为相对容差对接近 0 的指标（如 loss）会产生问题。

**新工程师易踩坑（Pitfall）：**

设置 `comparator="eq", threshold=0.85` 时，期望 `actual=0.8500000001` 通过是合理的（会通过）；但期望 `actual=0.8501` 通过则不会——`abs_tol=1e-6` 不是宽松的近似比较器，它只覆盖浮点舍入噪声。如果需要业务意义上的"足够接近"，应该用 `ge` + 略低的 threshold，而非 `eq` + 宽松期望。

---

### 4.5 `validator.py` — 运行时跨模型语义校验

**是什么：** `validate_output(json_str, version)` 对 VLM 输出 JSON 字符串做两轮检查：(1) JSON Schema 结构验证（`versions/v{version}/schema.json`），(2) `_extra_validations()` 业务规则检查（pet_present 一致性、action.distribution 求和=1.0、eating_metrics.speed 求和、primary action 与 distribution 最高概率对应、narrative 长度）。`CONFIDENCE_WARN_THRESHOLD = 0.5` 触发低置信度警告。

**为什么这样设计（Why）：**

Pydantic v2 的 `@field_validator` 和 `@model_validator` 只能做单模型内部约束（字段间一致性）。VLM 输出的跨字段语义约束（如 `pet_present=True → pet.action 必须非空`）涉及 JSON Schema 无法表达的逻辑，也不适合放进 Pydantic 模型（因为 VLM 输出是原始 JSON 字符串，不一定对应某个 Pydantic 模型）。`validate_output()` 填补这个空白：作为独立函数，显式调用，职责清晰。

**权衡（Tradeoff）：**

- 考虑过：用 `@model_validator(mode="after", context=...)` 把跨模型检查推进 Pydantic 模型，通过 context 传递关联数据（如 Sample 列表）。
- 为什么拒绝：这会把验证逻辑耦合到特定的生产者调用路径——只有 context 被正确传入时才能触发，漏传则静默通过，反而更危险。独立函数让调用方明确知道"我在做跨模型校验"。
- 2026-04-23 的修复提交（c6f90b4）在 Pydantic 模型层补充了 `@model_validator`，覆盖单模型的不变量。`validate_output()` 继续负责 JSON Schema + 业务规则这层。

**新工程师易踩坑（Pitfall）：**

`PetFeederEvent.model_validate(data)` 调用**不会**触发 `validate_output()` 里的业务规则检查——那是 VLM 输出专用的语义验证，不是 Pydantic 模型验证的一部分。直接入库路径（pet-data）如需语义检查，必须显式调用 `validate_output(json.dumps(data))`。

---

## 5. 扩展点

### 5.1 新增 Annotation 生产者类型

1. 在 `annotations.py` 中继承 `BaseAnnotation`，声明 `annotator_type: Literal["your_type"] = "your_type"` 和类型专有字段。
2. 将新类型加入 `Annotation` 联合类型（`LLMAnnotation | ClassifierAnnotation | ... | YourAnnotation`）。
3. 在 `__init__.py` 的 `__all__` 和 re-export 中添加新类。
4. 写测试（`tests/test_annotations.py` 或新文件），覆盖：(a) 正常构造通过，(b) extra 字段被 `extra="forbid"` 拒绝，(c) discriminator 正确分发到新类型。
5. 如果新类型需要数据库独立表：`alembic revision --autogenerate -m "add_your_type_table"`，提交**新的**迁移文件（不修改已有文件）。
6. PR 需要通过 schema_guard CI，下游 pet-annotation 的插件注册表更新是独立 PR。

### 5.2 新增 ModelCard 字段

1. 在 `ModelCard` 中添加字段（`new_field: SomeType | None = None`），加 `Field(description="...")`。
2. 如果字段需要持久化，生成新的 Alembic 迁移（`alembic revision --autogenerate`）——**已提交迁移不可修改**。
3. 在 `tests/test_model_card.py` 添加：默认行为测试 + 显式赋值测试。
4. 视影响范围，通知下游消费方（pet-infra / pet-eval / pet-quantize / pet-ota）。schema_guard CI 会自动提醒。

### 5.3 新增 VLM 输出 Schema 版本

1. 在 `src/pet_schema/versions/` 下创建新目录（如 `v1.1/`），放入 `schema.json`。
2. `validate_output(json_str, version="1.1")` 会自动找到新版本文件。
3. 旧版本文件保留（不删除），兼容历史 VLM 输出。

---

## 6. 依赖管理

### pet-schema 的依赖

pet-schema 无上游 `pet-*` 依赖。直接依赖：

| 包 | 版本约束 | 用途 |
|---|---|---|
| `pydantic` | `>=2.0,<3.0` | 数据模型核心 |
| `jsonschema` | `>=4.20,<5.0` | VLM 输出 JSON Schema 验证 |
| `jinja2` | `>=3.1,<4.0` | `render_prompt()` 模板渲染 |
| `networkx` | `>=3.2,<4.0` | `to_dag()` DAG 构造与环检测 |

可选依赖（`[adapters]`）：

| 包 | 版本约束 | 用途 |
|---|---|---|
| `datasets` | `>=2.19,<3.0` | `adapters/hf_features.py`（HuggingFace 集成） |
| `webdataset` | `>=0.2,<0.3` | WebDataset 格式支持 |

### 下游如何 pin pet-schema

当前生态存在 3 种 pin 风格（hardpin tag / no-pin / not-depending）。统一策略在 Phase 2（pet-infra pass）中决定——方案 α（hardpin 到版本 tag）vs 方案 β（peer-dep）仍 pending。

**(Phase 2 pin 决策 pending — backfill here after pet-infra pass)**

**临时规则（覆盖 Phase 2 决策前）：**
- 安装 pet-schema 必须 pin 到版本 tag（如 `pet-schema @ git+...@v2.4.0`），不能用 `@main`（CLAUDE.md 明确规定）。
- 如果下游 CI 因未 pin 而出现版本漂移，属于 P1 问题，不可用手动 workaround 绕过。

---

## 7. 本地开发与测试

### 环境准备

```bash
conda activate pet-pipeline     # 使用共享 pet-pipeline conda 环境，不单独建仓库环境
cd pet-schema
make setup                      # pip install -e ".[dev]"
```

### 常用命令

```bash
make test     # pytest tests/ -v --tb=short（当前 167 个测试）
make lint     # ruff + mypy
make clean    # 清理构建产物
```

### Alembic 注意事项

- **已提交的迁移文件不允许修改**（CLAUDE.md 强制规定）。
- 新建迁移：`alembic revision --autogenerate -m "描述"`。
- 生成的迁移文件需要人工检查（autogenerate 不总是完美），确认后提交。

### SCHEMA_VERSION 与 pyproject.toml 一致性

`version.py` 中的 `SCHEMA_VERSION` 必须与 `pyproject.toml` 的 `[project] version` 保持一致，`tests/test_version.py` 中的 `test_schema_version_matches_pyproject` 会自动检查这一点。如果 CI 报 `SCHEMA_VERSION` 不匹配，先检查两处是否同步更新。

### 常见 CI-only 失败模式

如果测试只在 CI 失败而本地通过，检查：
1. `SCHEMA_VERSION` parity（上述测试）。
2. `versions/` 目录下的 JSON Schema 文件是否在 `pyproject.toml` 的 `package-data` 中声明（`pet_schema = ["versions/**/*"]`）。
3. `[adapters]` 可选依赖是否需要安装（`make setup` 默认只装 `[dev]`，不含 HuggingFace `datasets`）。

---

## 8. 已知复杂点（复杂但必要）

### 8.1 RecipeStage.inputs + depends_on 双机制（finding #9）

**保留理由：** `inputs` 是有类型的制品绑定（orchestrator 知道"什么被传入"），`depends_on` 是 DAG 调度约束（orchestrator 知道"执行顺序"）。两者语义不同，合并会让 schema 层渗入 orchestration 逻辑。

**删了会损失什么：** 如果删掉 `depends_on` 只保留 `inputs`，DAG 环检测必须解析 `ArtifactRef.ref_type` 内部语义——schema 层要理解 orchestration 层的含义，职责混乱。如果删掉 `inputs` 只保留 `depends_on`，则制品的类型安全绑定消失，orchestrator 收到的输入变成无类型 dict。

**重新审视的触发条件：** 若未来出现新的 orchestration 范式，其中类型绑定本身足以推导全部 DAG 依赖，且这一推导无需解析 `ref_type` 语义（例如通过显式的 stage 引用对象替代字符串 name），可以考虑合并。当前无此需求。

---

### 8.2 ResourceSpec 不嵌入 ModelCard（finding #3 reclassified）

**保留理由：** `ResourceSpec` 是 trainer 插件的预飞接口（`TrainerBase.estimate_resources() → ResourceSpec`），在训练前声明期望资源。`ModelCard` 是训练后的规范描述。两者生命周期、产生时机和使用者完全不同。

**删了会损失什么：** trainer 插件失去类型安全的预飞资源声明接口；preflight 检查退化为无类型 dict 或启发式估算，orchestrator 无法在启动前做资源可用性判断。

**重新审视的触发条件：** 如果未来需要在 ModelCard 中记录实际资源消耗快照（区别于当前的 ClearML 追踪），可以在 ModelCard 加 `actual_resources: ResourceSpec | None = None` 字段；但这不等于删除独立的 `ResourceSpec`——预飞接口仍然需要。

---

### 8.3 PetFeederEvent legacy re-export（finding #12）

**保留理由：** `PetFeederEvent` 在 `__init__.py:39` 保留了顶层 re-export（`from pet_schema.models import PetFeederEvent  # legacy v1`）。经跨仓库 grep 确认（2026-04-23），pet-eval 在 `conftest.py` 和 `src/inference/constrained.py` 中直接使用 `from pet_schema import PetFeederEvent`。删除此 re-export 会立即破坏 pet-eval。

**删了会损失什么：** 需要先完成 pet-eval 从 `PetFeederEvent` 迁移到 `Sample + Annotation` 的跨仓库迁移（至少 2 个文件，可能涉及 pet-eval 的 DB schema 变更）。在迁移完成前删除是破坏性操作。

**重新审视的触发条件：** Phase 5 范围——当 pet-eval 将推理路径（`inference/constrained.py`）和测试 fixtures 迁移到 `Sample + Annotation` 后，在 `PetFeederEvent` 加 `DeprecationWarning`，并设定移除目标版本（建议 `v3.0.0` 配合 major bump）。

---

## 9. Phase 5+ Followups

### 9.1 PetFeederEvent validator 边界情况

**描述：** 当前 `_extra_validations()` 检查 `pet_present=True → pet 非空 + pet_count >= 1`，以及 `pet_present=False → pet 为 None`，但**未检查** `pet_present=False → pet_count == 0`。这个不变量在逻辑上是完整性要求，实现只需 3 行。

**为什么暂缓：** (a) 超出 finding #5 用户裁定的修复范围；(b) `Field(ge=0, le=4)` 限制了 `pet_count` 的爆炸半径；(c) VLM 生产者在实践中未观测到此矛盾状态。

**触发条件：** hardening pass 或真实生产事故（VLM 输出 `pet_present=False` 同时 `pet_count > 0`）。届时 3 行补丁即可合并。

---

### 9.2 CONFIDENCE_WARN_THRESHOLD 迁移到 params.yaml

**描述：** `validator.py:13` 的 `CONFIDENCE_WARN_THRESHOLD = 0.5` 是模块级常量，违反了 CLAUDE.md 的"所有数值从 params.yaml 读取"规则。常量的 docstring 已注明此迁移路径。这是跨切面变更：需要 DEVELOPMENT_GUIDE（pet-infra）先明确 schema-layer 数值的 params.yaml 约定（当前 params.yaml 约定只覆盖 trainer/eval 层数值）。

**触发条件：** pet-infra Phase 2 pass 明确 params.yaml 约定扩展到 schema 层之后。届时：(1) 更新 DEVELOPMENT_GUIDE，(2) 将 `CONFIDENCE_WARN_THRESHOLD` 改为从 params.yaml 读取，(3) 同步 pet-schema 版本 bump。

---

### 9.3 DEVELOPMENT_GUIDE 补充 schema-layer 硬编码阈值说明

**描述：** DEVELOPMENT_GUIDE（`pet-infra/docs/DEVELOPMENT_GUIDE.md`）应明确说明 schema-layer 阈值（`CONFIDENCE_WARN_THRESHOLD`, `GateCheck.abs_tol=1e-6` 等）目前为硬编码，并文档化其迁移到 params.yaml 的路径。这是跨仓库文档同步，不是 pet-schema 代码变更。

**触发条件：** pet-infra Phase 2 pass 中处理 DEVELOPMENT_GUIDE 时顺带完成。不需要单独 PR。

---

### 9.4 test_adapters_manifest.py 硬编码版本号

**描述：** `tests/test_adapters_manifest.py:36` 中有 `assert m["schema_version"] == "2.4.0"`（硬编码）。这与 `test_schema_version_matches_constant`（line 46-49，使用 `SCHEMA_VERSION` import 做 parity 检查）存在冗余。下次 `SCHEMA_VERSION` bump 时如果忘记更新这行，测试会误报。

**触发条件：** 下次 `SCHEMA_VERSION` bump 时若该测试因此失败，顺手将 `"2.4.0"` 替换为 `from pet_schema import SCHEMA_VERSION` import。低优先级，已被 parity 测试兜底。
