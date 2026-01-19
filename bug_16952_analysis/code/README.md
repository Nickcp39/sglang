# Bug #16952 修复代码

## 文件说明

### 修复代码文件

- **`fix_server_args.py`**: `server_args.py` 的修复代码说明
- **`fix_ep_moe_layer.py`**: `ep_moe/layer.py` 的修复代码说明

### Git Patch 文件

- **`patch_server_args.py.patch`**: `server_args.py` 的 Git patch
- **`patch_ep_moe_layer.py.patch`**: `ep_moe/layer.py` 的 Git patch

## 应用修复的方法

### 方法 1: 使用 Git Patch（推荐）

```powershell
# 在项目根目录执行
cd d:\Yc-software\github_code\sglang

# 应用 patch
git apply bug_16952_analysis/code/patch_server_args.py.patch
git apply bug_16952_analysis/code/patch_ep_moe_layer.py.patch
```

### 方法 2: 手动应用

查看 `fix_server_args.py` 和 `fix_ep_moe_layer.py` 文件，按照说明手动修改代码。

### 方法 3: 使用脚本

```powershell
cd bug_16952_analysis\test
python apply_fix.py
```

**注意**: 脚本会检查修复状态，但自动应用功能需要手动实现。

## 修复内容

### 1. server_args.py 修复

**位置**: `_handle_model_specific_adjustments()` 方法末尾（约第 797 行后）

**功能**: 当启用 TBO/SBO + DeepEP 时，自动选择 `flashinfer_cutedsl` backend

**关键代码**:
```python
# Fix for Bug #16952: Auto-select flashinfer_cutedsl when TBO/SBO + DeepEP is enabled
if self.enable_two_batch_overlap or self.enable_single_batch_overlap:
    if self.moe_a2a_backend == "deepep":
        if self.moe_runner_backend == "auto" or self.moe_runner_backend is None:
            # 自动选择逻辑
            ...
```

### 2. ep_moe/layer.py 修复

**位置**: `moe_impl()` 方法中（约第 489-495 行）

**功能**: 双重检查，防止使用已弃用的 `forward_deepgemm_masked`

**关键代码**:
```python
# Fix for Bug #16952: Double-check to prevent using deprecated forward_deepgemm_masked
if down_gemm_overlap_args is not None or is_tbo_enabled():
    raise RuntimeError(
        "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. "
        "Please use --moe-runner-backend flashinfer_cutedsl."
    )
```

## 验证修复

### 检查修复是否已应用

```powershell
# 检查 server_args.py
Select-String -Path python\sglang\srt\server_args.py -Pattern "TBO/SBO is enabled with DeepEP"

# 检查 layer.py
Select-String -Path python\sglang\srt\layers\moe\ep_moe\layer.py -Pattern "TBO/SBO requires --moe-runner-backend"
```

**预期输出**: 应该找到匹配的行

## 恢复原始代码

如果需要恢复：

```powershell
# 使用备份文件
Copy-Item python\sglang\srt\server_args.py.backup python\sglang\srt\server_args.py
Copy-Item python\sglang\srt\layers\moe\ep_moe\layer.py.backup python\sglang\srt\layers\moe\ep_moe\layer.py

# 或使用 Git
git checkout python/sglang/srt/server_args.py
git checkout python/sglang/srt/layers/moe/ep_moe/layer.py
```
