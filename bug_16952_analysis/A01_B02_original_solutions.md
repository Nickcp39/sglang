# A01_B02: Bug #16952 原始解决方案汇总

## 📋 目录
1. [用户原始解决方案（Nickcp39）](#用户原始解决方案nickcp39)
2. [Fridge003 的解决方案](#fridge003-的解决方案)
3. [两种方案的关系](#两种方案的关系)

## 相关文档
- [A01: flashinfer_cutedsl 详解](./A01_flashinfer.md) - 了解 flashinfer_cutedsl 是什么
- [A01_B01: 修复前后代码对比](./A01_B01_fix_comparison.md) - 修复前后代码对比与详细解释

---

## 用户原始解决方案（Nickcp39）

### Issue Comment

```
Hi @ziyuhuang123,

Quick workaround (you can test immediately):
Add --moe-runner-backend flashinfer_cutedsl to your command.
This should resolve the error.

Root Cause:
TBO requires flashinfer_cutedsl backend, but when --moe-runner-backend is auto, 
the code falls back to forward_deepgemm_masked which doesn't support TBO.

Permanent Fix (in progress):
I'm working on a fix that will automatically select flashinfer_cutedsl when 
TBO/SBO + DeepEP is enabled, so users won't need to remember this parameter.

PR coming soon!

Please test the workaround and let me know if it works.
```

### 解决方案说明

#### Workaround（临时方案）
```bash
# 用户需要手动添加参数
python3 -m sglang.launch_server \
    --model-path /dev/shm/qwen30b/qwen30b \
    --tp 8 --ep 8 \
    --enable-two-batch-overlap \
    --moe-a2a-backend deepep \
    --moe-runner-backend flashinfer_cutedsl  # ✅ 手动指定
```

#### Permanent Fix（永久修复）
- **位置**：`python/sglang/srt/server_args.py`
- **方法**：`_handle_model_specific_adjustments()`
- **逻辑**：检测到 TBO/SBO + DeepEP + auto → 自动选择 `flashinfer_cutedsl`

#### 代码实现

```python
# python/sglang/srt/server_args.py:799-821

# Fix for Bug #16952: Auto-select flashinfer_cutedsl when TBO/SBO + DeepEP is enabled
if self.enable_two_batch_overlap or self.enable_single_batch_overlap:
    if self.moe_a2a_backend == "deepep":
        if self.moe_runner_backend == "auto" or self.moe_runner_backend is None:
            # Check if flashinfer_cutedsl is available
            try:
                from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
                self.moe_runner_backend = "flashinfer_cutedsl"
                logger.info(
                    "TBO/SBO is enabled with DeepEP. "
                    "Automatically using flashinfer_cutedsl backend for overlap optimization."
                )
            except ImportError:
                raise RuntimeError(
                    "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl, "
                    "but flashinfer_cutedsl is not available. "
                    "Please install flashinfer with cutedsl support."
                )
        elif self.moe_runner_backend != "flashinfer_cutedsl":
            raise RuntimeError(
                "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. "
                "Please use --moe-runner-backend flashinfer_cutedsl."
            )
```

#### Layer 层面的双重检查

```python
# python/sglang/srt/layers/moe/ep_moe/layer.py:497-501

# Fix for Bug #16952: Double-check to prevent using deprecated forward_deepgemm_masked
# when TBO/SBO is enabled
if down_gemm_overlap_args is not None or is_tbo_enabled():
    raise RuntimeError(
        "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. "
        "Please use --moe-runner-backend flashinfer_cutedsl."
    )
```

---

## Fridge003 的解决方案

### Issue Comment

```
@ziyuhuang123 This bug happens when you try to applies DeepEP+TBO on a bfloat16 
model checkpoint. You can try fp8 qwen checkpoint instead
```

### 解决方案说明

#### 核心观点
- **问题根源**：Bug 发生在使用 **bfloat16 checkpoint** 时
- **解决方案**：改用 **fp8 checkpoint**

#### 为什么有效？

1. **fp8 checkpoint 的特性**：
   - fp8 checkpoint 使用 `ModelOptNvFp4FusedMoEMethod`
   - `ModelOptNvFp4FusedMoEMethod` 有 `apply_without_routing_weights` 方法
   - 这个方法**强制要求** `flashinfer_cutedsl` backend

2. **代码证据**：
   ```python
   # python/sglang/srt/layers/quantization/modelopt_quant.py:1478
   
   def apply_without_routing_weights(...):
       assert self.enable_flashinfer_cutedsl_moe, "only support flashinfer cutedsl moe"
       # ⚠️ 如果 backend 不是 flashinfer_cutedsl，这里会 assert 失败
   ```

3. **结果**：
   - 使用 fp8 checkpoint 时，用户**必须**设置 `--moe-runner-backend flashinfer_cutedsl`
   - 否则 `apply_without_routing_weights` 会 assert 失败
   - 因此，fp8 checkpoint 用户**天然**会使用正确的 backend

#### 使用方式

```bash
# 使用 fp8 checkpoint（而不是 bfloat16）
python3 -m sglang.launch_server \
    --model-path /path/to/qwen30b_fp8 \  # ⚠️ fp8 checkpoint
    --tp 8 --ep 8 \
    --enable-two-batch-overlap \
    --moe-a2a-backend deepep \
    --moe-runner-backend flashinfer_cutedsl  # ⚠️ 必须指定（否则 assert 失败）
```

---

## 两种方案的关系

### 方案对比

| 方面 | 用户方案（Nickcp39） | Fridge003 方案 |
|------|---------------------|----------------|
| **解决层面** | 代码层面（自动选择） | 数据层面（更换 checkpoint） |
| **适用场景** | 所有 checkpoint 类型 | 只适用于可以更换 checkpoint 的情况 |
| **用户操作** | 无需操作（自动）或手动添加参数 | 需要更换 checkpoint |
| **根本性** | ✅ 解决根本问题 | ⚠️ 绕过问题（如果用户只能用 BF16 checkpoint） |

### 两种方案的关系

#### 互补关系

1. **用户方案**：
   - ✅ 解决**代码层面**的问题
   - ✅ 让 bfloat16 checkpoint 也能自动选择正确的 backend
   - ⚠️ 但如果 bfloat16 checkpoint 不支持 `flashinfer_cutedsl`，仍然会出错

2. **Fridge003 方案**：
   - ✅ 解决**数据层面**的问题
   - ✅ 使用 fp8 checkpoint 天然支持 `flashinfer_cutedsl`
   - ⚠️ 需要用户更换 checkpoint（如果用户只能用 BF16，无法解决）

#### 组合使用

- **最佳实践**：
   - 如果可以使用 fp8 checkpoint → 使用 Fridge003 的方案
   - 如果必须使用 bfloat16 checkpoint → 使用用户的方案（自动选择）
   - **两者结合**：用户的代码修复 + fp8 checkpoint = 最稳定的方案

### 为什么两种方案都有效？

#### 用户方案的有效性
- **原因**：在代码层面自动选择 `flashinfer_cutedsl`
- **前提**：checkpoint 必须支持 `flashinfer_cutedsl`（fp4/fp8）
- **限制**：如果 checkpoint 不支持（BF16），仍然会出错

#### Fridge003 方案的有效性
- **原因**：fp8 checkpoint 强制要求 `flashinfer_cutedsl` backend
- **前提**：用户可以使用 fp8 checkpoint
- **限制**：如果用户只能用 BF16 checkpoint，无法使用此方案

---

## 总结

### 用户方案（Nickcp39）
- **类型**：代码修复（自动选择 backend）
- **优点**：解决根本问题，用户无需手动指定参数
- **限制**：需要 checkpoint 支持 `flashinfer_cutedsl`

### Fridge003 方案
- **类型**：数据层面解决方案（更换 checkpoint）
- **优点**：fp8 checkpoint 天然支持，更稳定
- **限制**：需要用户更换 checkpoint

### 最佳实践
- **优先**：使用 fp8 checkpoint（Fridge003 方案）
- **补充**：代码层面自动选择（用户方案）
- **组合**：两者结合使用，最稳定可靠
