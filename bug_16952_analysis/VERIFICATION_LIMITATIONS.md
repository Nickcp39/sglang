# Bug #16952 验证限制说明

## ⚠️ 关键问题

**我们无法在自己的 RTX 4090 环境上验证"加参数就能解决 bug"这个理论。**

---

## 为什么无法验证？

### 1. 硬件限制

**RTX 4090（SM 8.9）的限制**：
- ❌ 不支持 `deep_gemm`（需要 SM >= 90）
- ❌ 必须使用 `--deepep-mode normal`
- ❌ `--deepep-mode normal` 会自动禁用 CUDA Graph

### 2. Bug 发生的位置

**原始 Bug 发生在**：
- CUDA Graph capture 阶段
- 具体位置：`cuda_graph_runner.py` → `capture()` → `forward()` → `ep_moe/layer.py` → `run_moe_core()`

### 3. 无法复现的原因

**即使加了 `--moe-runner-backend flashinfer_cutedsl`**：
- ✅ 参数会被正确解析
- ✅ 自动选择逻辑会工作
- ✅ 服务器可以启动
- ❌ **但无法进入 CUDA Graph capture 阶段**（因为 CUDA Graph 被禁用）
- ❌ **因此无法验证"是否真的解决了原始 Bug"**

---

## 我们能验证什么？

### ✅ 可以验证（RTX 4090）

1. **自动选择逻辑**：
   - 修复后，系统会自动选择 `flashinfer_cutedsl`
   - 可以看到日志：`TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend...`

2. **双重检查逻辑**：
   - 如果配置错误，会抛出清晰的 `RuntimeError`
   - 错误信息：`TBO/SBO requires --moe-runner-backend flashinfer_cutedsl...`

3. **服务器启动**：
   - 修复后，服务器可以正常启动（在 `--deepep-mode normal` 模式下）

### ❌ 无法验证（RTX 4090）

1. **CUDA Graph capture 阶段**：
   - 无法进入 capture 阶段（CUDA Graph 被禁用）
   - 无法验证"加参数是否真的解决了原始 Bug"

2. **完整的修复验证**：
   - 需要 SM>=90 GPU（H100/H20/H200）
   - 需要启用 CUDA Graph（`deepep_mode='auto'` 或 `'low_latency'`）

---

## 验证策略

### 1. 代码分析验证（已完成）

✅ **通过代码逻辑确认**：
- TBO 需要 `flashinfer_cutedsl` backend
- `forward_deepgemm_masked` 不支持 TBO
- 加参数应该能解决问题

### 2. 原始 Issue 报告者验证（需要）

⚠️ **需要 @ziyuhuang123 验证**：
- 他们有 H20 GPU（SM 9.0）
- 可以启用 CUDA Graph
- 可以验证 workaround 是否有效

**建议回复**：
```markdown
**Note**: I cannot verify this workaround on my hardware (RTX 4090, SM 8.9) because it doesn't support CUDA Graph capture where the bug occurs. Since you have H20 GPUs (SM 9.0), could you please test this workaround and confirm if it resolves the issue?
```

### 3. Maintainer/贡献者验证（PR 后）

⚠️ **需要 SM>=90 GPU 验证**：
- 完整的 CUDA Graph capture 验证
- 修复前后的对比测试
- 性能测试（可选）

---

## 在 Issue 回复中如何说明？

### 推荐方式

```markdown
**Quick Workaround** (you can test this immediately):

Add `--moe-runner-backend flashinfer_cutedsl` to your command.

**Note**: I cannot verify this workaround on my hardware (RTX 4090, SM 8.9) because it doesn't support CUDA Graph capture where the bug occurs. Since you have H20 GPUs (SM 9.0), could you please test this workaround and confirm if it resolves the issue?

**Permanent Fix** (in progress):
I'm working on a fix that will automatically select `flashinfer_cutedsl` when TBO/SBO + DeepEP is enabled, so users won't need to remember this parameter.
```

### 为什么这样说明？

1. **诚实透明**：说明我们的限制
2. **请求帮助**：让原始报告者验证 workaround
3. **展示价值**：说明我们的修复（自动化）仍然有价值

---

## 总结

### 我们能做的

1. ✅ **代码分析**：确认理论正确
2. ✅ **逻辑验证**：验证自动选择和双重检查逻辑
3. ✅ **部分验证**：在 RTX 4090 上验证服务器启动

### 我们无法做的

1. ❌ **完整验证**：无法验证"加参数是否真的解决了原始 Bug"
2. ❌ **CUDA Graph 验证**：无法进入 capture 阶段

### 需要帮助的

1. ⚠️ **原始 Issue 报告者**：验证 workaround
2. ⚠️ **Maintainer/贡献者**：验证完整的修复（SM>=90 GPU）

---

## 建议

1. **在 Issue 回复中说明限制**：诚实透明
2. **请求原始报告者验证**：他们有正确的硬件
3. **在 PR 中说明**：明确哪些已验证，哪些需要 maintainer 验证
4. **提供测试清单**：让 maintainer 知道需要验证什么

这样既诚实，又不会让 maintainer 觉得"你没测试就提交了"。
