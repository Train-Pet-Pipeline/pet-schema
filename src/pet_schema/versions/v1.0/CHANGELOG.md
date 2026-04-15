# CHANGELOG - Schema v1.0

## v1.0 (2026-04-14)

初始版本。

### 与 DEVELOPMENT_GUIDE 原版的差异

- `body_signals.posture`: 从 3 值概率分布改为枚举（含 unobservable）
- `body_signals.ear_position`: 从 4 值概率分布改为枚举（含 unobservable）
- `mood.comfort`: 删除（与 anxiety 负相关冗余）
- `narrative.maxLength`: 从 50 放宽到 80
