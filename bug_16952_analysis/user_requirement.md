# Bug #16952 用户需求分析

## 用户的原始表述

> "I can run qwen30b **without tbo and deepep**. But **using tbo and deepep**, it reports bug."

## 关键信息提取

### 1. "without tbo and deepep" - 可以运行

**含义**:
- 用户知道这两个选项是什么
- 用户知道不使用这两个选项时，代码可以正常工作
- 这说明用户对这两个功能有基本了解

### 2. "using tbo and deepep" - 报错

**含义**:
- 用户**明确想要同时使用** TBO 和 DeepEP
- 用户认为这是一个**合理的组合**
- 用户遇到了错误

### 3. "it reports bug" - 用户认为这是 bug

**含义**:
- 用户不认为这是配置错误
- 用户认为这应该是一个可以工作的组合
- 用户期望修复后能够工作

## 用户的命令

```bash
python3 -m sglang.launch_server \
    --model-path /dev/shm/qwen30b/qwen30b \
    --tp 8 --ep 8 \
    --port 32123 \
    --trust-remote-code \
    --enable-dp-attention \
    --disable-custom-all-reduce \
    --enable-two-batch-overlap \      # ✅ 用户启用了 TBO
    --moe-a2a-backend deepep          # ✅ 用户启用了 DeepEP
    # ❌ 但没有指定 --moe-runner-backend flashinfer_cutedsl
```

## 用户需求分析

### 核心需求

**用户明确想要**:
1. ✅ **使用 TBO** (`--enable-two-batch-overlap`)
2. ✅ **使用 DeepEP** (`--moe-a2a-backend deepep`)
3. ✅ **运行 Qwen3-30B 模型**
4. ✅ **让这个组合能够工作**

### 用户可能不知道的事情

1. **需要额外的 backend 配置**
   - 用户可能不知道需要 `--moe-runner-backend flashinfer_cutedsl`
   - 用户可能认为 `--moe-a2a-backend deepep` 就足够了

2. **TBO + DeepEP 的技术依赖**
   - 用户可能不知道 TBO 需要特定的 backend 支持
   - 用户可能认为只要启用了这两个选项就能工作

### 用户的期望

1. **TBO + DeepEP 应该是一个可以工作的组合**
   - 用户同时启用了这两个选项
   - 用户期望它们能够协同工作

2. **用户认为这是代码的问题，不是配置问题**
   - 用户说"reports bug"，而不是"configuration error"
   - 用户期望修复后能够工作

3. **如果配置不完整，期望得到清晰的错误信息**
   - 用户报告 bug，期望得到修复
   - 清晰的错误信息可以帮助用户快速解决问题

## 修复方案选择

### 方案 A: 直接报错（推荐）

**理由**:
1. **用户明确想要使用 TBO + DeepEP**
   - 用户说"using tbo and deepep"，说明用户想要这个组合
   - 如果配置不完整，应该明确告诉用户需要什么

2. **用户认为这是 bug，期望修复**
   - 用户报告 bug，期望得到修复
   - 清晰的错误信息可以帮助用户快速解决问题

3. **防止用户误以为 TBO 在工作**
   - 如果静默降级，用户可能不知道 TBO 实际上没有工作
   - 用户可能以为性能优化生效了，但实际上没有

4. **与现有代码设计一致**
   - 代码中已经有 `assert False, "forward_deepgemm_masked is deprecated"`
   - 这说明代码设计者认为这种情况应该报错

### 方案 B: 警告 + 优雅降级（不推荐）

**为什么不推荐**:
1. **不符合用户期望**
   - 用户明确想要使用 TBO，如果静默降级，用户可能不知道 TBO 没有工作
   - 用户报告 bug，期望修复，而不是降级

2. **可能隐藏问题**
   - 用户可能以为性能优化生效了，但实际上没有
   - 用户可能不知道为什么性能没有提升

## 关键发现

### 模型本身支持 TBO + DeepEP

**证据**:
1. **测试用例存在**: `test/srt/test_two_batch_overlap.py` 中有 `TestQwen3TwoBatchOverlap` 测试类
2. **测试配置**: 测试中使用了 `--enable-two-batch-overlap` + `--moe-a2a-backend deepep`
3. **文档说明**: `--enable-dp-attention` 的帮助信息明确说明 "Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported"

### 问题不是模型不支持，而是配置不完整

**用户的配置**:
```bash
--enable-two-batch-overlap      # ✅ 正确
--moe-a2a-backend deepep       # ✅ 正确
# ❌ 缺少：--moe-runner-backend flashinfer_cutedsl
```

**正确的配置应该是**:
```bash
--enable-two-batch-overlap
--moe-a2a-backend deepep
--moe-runner-backend flashinfer_cutedsl  # ← 缺少这个
```

### 为什么需要 `--moe-runner-backend flashinfer_cutedsl`？

**技术原因**:
- TBO 需要 overlap 功能（SM 控制、信号同步等）
- 只有 `flashinfer_cutedsl` backend 的 kernel 支持这些功能
- `forward_deepgemm_masked` 使用的 kernel 不支持这些功能

**代码证据** (`single_batch_overlap.py:20-26`):
```python
@classmethod
def enable_combine_down_gemm_two_stream_overlap(cls):
    return (
        is_sbo_enabled()
        # currently only cutedsl backend supports it  ← 明确说明
        and get_moe_runner_backend().is_flashinfer_cutedsl()
    )
```

## 最终推荐

**推荐方案 A（直接报错）**:
- ✅ 符合用户期望（用户想要使用 TBO，应该明确告诉用户需要什么配置）
- ✅ 防止用户误以为 TBO 在工作
- ✅ 与现有代码设计一致
- ✅ 帮助用户快速解决问题

**修复后的效果**:
- 用户会看到清晰的错误信息，知道需要添加 `--moe-runner-backend flashinfer_cutedsl`
- 添加后，TBO + DeepEP 就能正常工作（因为模型本身支持）
