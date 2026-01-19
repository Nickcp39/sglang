# Issue #16952 回复模板

## 选项 1：临时 Workaround（快速解决）

如果你只是想快速解决用户的问题，可以这样回复：

---

### 回复内容

```markdown
Hi @ziyuhuang123,

Thanks for reporting this issue! I've identified the root cause.

**Quick Workaround** (you can test this immediately):

Add `--moe-runner-backend flashinfer_cutedsl` to your command:

```bash
python3 -m sglang.launch_server \
    --model-path /dev/shm/qwen30b/qwen30b \
    --tp 8 --ep 8 \
    --port 32123 \
    --trust-remote-code \
    --enable-dp-attention \
    --disable-custom-all-reduce \
    --enable-two-batch-overlap \
    --moe-a2a-backend deepep \
    --moe-runner-backend flashinfer_cutedsl  # ← Add this parameter
```

This should resolve the `AssertionError: forward_deepgemm_masked is deprecated` error.

**Root Cause**:
- TBO (Two Batch Overlap) requires `flashinfer_cutedsl` backend
- When `--moe-runner-backend` is `auto` or not specified, the code falls back to `forward_deepgemm_masked`, which doesn't support TBO

**Note**: I cannot verify this workaround on my hardware (RTX 4090, SM 8.9) because it doesn't support CUDA Graph capture where the bug occurs. Since you have H20 GPUs (SM 9.0), could you please test this workaround and confirm if it resolves the issue?

**Permanent Fix** (in progress):
I'm working on a fix that will automatically select `flashinfer_cutedsl` when TBO/SBO + DeepEP is enabled, so users won't need to remember this parameter.

Please let me know if the workaround works for you!
```

---

## 选项 2：完整修复说明（推荐）

如果你已经完成了修复，可以这样回复：

---

### 回复内容

```markdown
Hi @ziyuhuang123,

Thanks for reporting this issue! I've identified and fixed the root cause.

**Root Cause**:
- TBO (Two Batch Overlap) requires `flashinfer_cutedsl` backend
- When `--moe-runner-backend` is `auto` or not specified, the code falls back to `forward_deepgemm_masked`, which doesn't support TBO and triggers the `AssertionError`

**Fix**:
I've implemented a fix that:
1. **Automatically selects** `flashinfer_cutedsl` when TBO/SBO + DeepEP is enabled (so you don't need to remember the parameter)
2. **Provides clear error messages** if the configuration is incorrect

**Before the fix** (your original command):
```bash
--enable-two-batch-overlap \
--moe-a2a-backend deepep
# ❌ Missing: --moe-runner-backend flashinfer_cutedsl
# Result: AssertionError: forward_deepgemm_masked is deprecated
```

**After the fix** (same command, no changes needed):
```bash
--enable-two-batch-overlap \
--moe-a2a-backend deepep
# ✅ Automatically selects flashinfer_cutedsl
# Result: Works correctly
```

**Quick Workaround** (if you need it before the fix is merged):
Add `--moe-runner-backend flashinfer_cutedsl` to your command.

**Note**: I cannot verify this workaround on my hardware (RTX 4090, SM 8.9) because it doesn't support CUDA Graph capture where the bug occurs. Since you have H20 GPUs (SM 9.0), could you please test this workaround and confirm if it resolves the issue?

I'll create a PR shortly. Please let me know if you'd like to test the fix!
```

---

## 选项 3：简洁版（如果你不确定）

如果你不确定修复是否完成，可以这样回复：

---

### 回复内容

```markdown
Hi @ziyuhuang123,

I've identified the root cause. TBO requires `flashinfer_cutedsl` backend, but when `--moe-runner-backend` is `auto`, the code falls back to `forward_deepgemm_masked` which doesn't support TBO.

**Quick workaround** (please test):
Add `--moe-runner-backend flashinfer_cutedsl` to your command.

**Note**: I cannot verify this workaround on my hardware (RTX 4090, SM 8.9) because it doesn't support CUDA Graph capture where the bug occurs. Since you have H20 GPUs (SM 9.0), could you please test this workaround and confirm if it resolves the issue?

I'm working on a fix to automatically select the correct backend. Will update here once the PR is ready.
```

---

## 建议

### 如果你已经完成修复代码

**推荐使用选项 2**（完整修复说明）：
- ✅ 说明你已经修复了问题
- ✅ 解释修复的价值（自动化）
- ✅ 提供 workaround 作为备选

### 如果你还在测试修复

**推荐使用选项 1**（临时 Workaround）：
- ✅ 快速解决用户的问题
- ✅ 说明你正在修复
- ✅ 让用户知道这不是最终方案

### 如果你不确定

**推荐使用选项 3**（简洁版）：
- ✅ 提供 workaround
- ✅ 说明你在工作
- ✅ 保持简洁

---

## 注意事项

1. **不要只说"加参数"**：
   - 用户报告了 bug，期望的是"修复"
   - 只说 workaround 可能让用户觉得"这不是真正的修复"

2. **说明修复的价值**：
   - 自动化比手动加参数更好
   - 这是真正的"修复"，不是 workaround

3. **提供测试机会**：
   - 如果修复已完成，邀请用户测试
   - 如果还在进行中，说明进度

---

## 我的建议

**如果你已经完成了修复代码**，我建议：

1. **先回复选项 2**（完整修复说明）
2. **同时提供 workaround**（让用户可以先测试）
3. **说明 PR 即将提交**（让用户知道进度）

这样既解决了用户的燃眉之急，又展示了你的修复价值。
