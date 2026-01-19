# Bug #16952 测试说明

## 文件说明

- `apply_fix.py`: 应用修复代码到实际文件
- `test_fix.py`: 运行测试验证修复

## 使用步骤

### 1. 应用修复

```powershell
cd d:\Yc-software\github_code\sglang\bug_16952_analysis\test
python apply_fix.py
```

这会：
- 备份原始文件（添加 `.backup` 扩展名）
- 应用修复代码到 `server_args.py` 和 `ep_moe/layer.py`

### 2. 运行测试

```powershell
python test_fix.py
```

这会运行三个测试：
1. **Test 1**: 自动选择 flashinfer_cutedsl（应该成功）
2. **Test 2**: 显式指定 flashinfer_cutedsl（应该成功）
3. **Test 3**: 使用错误的 backend（应该失败并显示清晰的错误信息）

### 3. 手动测试

也可以手动运行测试命令：

```powershell
# 测试 1：自动选择（应该成功）
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep

# 测试 2：显式指定（应该成功）
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --moe-runner-backend flashinfer_cutedsl

# 测试 3：错误配置（应该失败）
python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 2 --ep 2 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --moe-runner-backend triton
```

## 恢复原始文件

如果需要恢复原始文件：

```powershell
# 恢复 server_args.py
Copy-Item python\sglang\srt\server_args.py.backup python\sglang\srt\server_args.py

# 恢复 ep_moe/layer.py
Copy-Item python\sglang\srt\layers\moe\ep_moe\layer.py.backup python\sglang\srt\layers\moe\ep_moe\layer.py
```
