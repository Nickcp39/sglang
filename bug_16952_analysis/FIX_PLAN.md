# Bug #16952 修复计划

## 问题描述

用户运行 Qwen3-30B 时，使用 `--enable-two-batch-overlap` (TBO) 和 `--moe-a2a-backend deepep` 时，遇到 `AssertionError: forward_deepgemm_masked is deprecated`。

**用户原始表述**：
> "I can run qwen30b without tbo and deepep. But using tbo and deepep, it reports bug."

**复现命令**：
```bash
python3 -m sglang.launch_server \
    --model-path /dev/shm/qwen30b/qwen30b \
    --tp 8 --ep 8 \
    --port 32123 \
    --trust-remote-code \
    --enable-dp-attention \
    --disable-custom-all-reduce \
    --enable-two-batch-overlap \
    --moe-a2a-backend deepep
```

## 根本原因

1. **用户配置不完整**：启用了 TBO + DeepEP，但没有指定 `--moe-runner-backend flashinfer_cutedsl`
2. **代码回退到废弃函数**：`DeepEPMoE.moe_impl` 在 `flashinfer_cutedsl` 未选择时，回退到 `forward_deepgemm_masked`
3. **废弃函数不支持 TBO**：`forward_deepgemm_masked` 不支持 overlap 功能，导致 AssertionError

## 修复策略

### 方案：自动选择 + 双重检查

1. **自动选择**（`server_args.py`）：
   - 当 TBO/SBO + DeepEP 启用时，如果 `moe_runner_backend` 为 `auto` 或 `None`，自动选择 `flashinfer_cutedsl`
   - 如果 `flashinfer_cutedsl` 不可用，给出清晰的错误信息

2. **双重检查**（`ep_moe/layer.py`）：
   - 在 `moe_impl` 中，如果检测到 TBO/SBO 启用但使用了错误的 backend，直接报错
   - 提供清晰的错误信息，指导用户使用正确的配置

## 代码修改

### 1. `python/sglang/srt/server_args.py`

在 `_handle_model_specific_adjustments` 方法中添加：

```python
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

### 2. `python/sglang/srt/layers/moe/ep_moe/layer.py`

在 `moe_impl` 方法中添加双重检查：

```python
# Fix for Bug #16952: Double-check to prevent using deprecated forward_deepgemm_masked
# when TBO/SBO is enabled
if down_gemm_overlap_args is not None or is_tbo_enabled():
    raise RuntimeError(
        "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. "
        "Please use --moe-runner-backend flashinfer_cutedsl."
    )
```

## 测试计划

### ⚠️ 重要：使用 Docker 进行测试

**强烈建议使用 Docker**，原因：
1. DeepEP 安装复杂，Docker 镜像已预装
2. NCCL 版本正确，避免通信错误
3. 复现一致性，与用户环境一致

详见：[DOCKER_TESTING_GUIDE.md](./DOCKER_TESTING_GUIDE.md)

### 测试场景

#### 场景 1: 复现 Bug（修复前）
```bash
# 应该看到 AssertionError
docker run --gpus all --shm-size 32g -p 32123:32123 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --network=host --privileged \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tp 2 --ep 2 --port 32123 \
        --trust-remote-code \
        --enable-dp-attention \
        --disable-custom-all-reduce \
        --enable-two-batch-overlap \
        --moe-a2a-backend deepep \
        --log-level info
```

#### 场景 2: 验证修复（修复后）
```bash
# 应该看到自动选择日志，然后成功启动
# 预期日志：
# "TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend..."
```

#### 场景 3: 手动指定 backend（验证兼容性）
```bash
# 手动指定 flashinfer_cutedsl，应该也能工作
--moe-runner-backend flashinfer_cutedsl
```

### 测试脚本

使用 PowerShell 脚本（Windows）：
```powershell
cd bug_16952_analysis\test
.\docker_test.ps1
```

### 硬件要求

- **RTX 4090**：可以使用较小的模型（Qwen1.5-MoE-A2.7B-Chat）进行测试
- **Qwen3-30B**：需要 8 个 GPU（`--tp 8 --ep 8`），不适合单卡测试

## 影响分析

### 向后兼容性
- ✅ **完全兼容**：只影响 TBO/SBO + DeepEP 的组合
- ✅ **其他场景不受影响**：没有 TBO/SBO 时，行为不变

### 全局影响
- ✅ **影响范围小**：只修改两个文件
- ✅ **逻辑清晰**：自动选择在参数解析阶段，双重检查在运行时
- ✅ **错误信息清晰**：用户能快速理解问题

## 预期效果

修复后：
1. ✅ 用户不需要手动指定 `--moe-runner-backend flashinfer_cutedsl`
2. ✅ 系统自动选择正确的 backend
3. ✅ 如果配置错误，给出清晰的错误信息
4. ✅ TBO + DeepEP 组合可以正常工作

## 相关文档

- [用户需求分析](./user_requirement.md)
- [Docker 测试指南](./DOCKER_TESTING_GUIDE.md)
- [模型下载说明](./MODEL_DOWNLOAD_EXPLAINED.md)
- [RTX 4090 测试指南](./rtx4090_testing_guide.md)
