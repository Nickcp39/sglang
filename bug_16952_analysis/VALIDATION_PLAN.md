# Bug #16952 验证计划

## ⚠️ 重要限制

**在 RTX 4090 上无法复现原始 Bug**，因为：
- RTX 4090（SM 8.9）不支持 `deep_gemm`
- 必须使用 `--deepep-mode normal`，这会禁用 CUDA Graph
- Bug #16952 发生在 CUDA Graph capture 阶段
- 因此 Bug 不会触发

**但是，我们仍然可以验证修复代码的逻辑是否正确！**

---

## ✅ 可以验证的内容（在 RTX 4090 上）

### 1. 验证自动选择逻辑（修复前 vs 修复后）

#### 修复前（预期行为）
- 使用 `--moe-runner-backend auto`（默认）
- 不会自动选择 `flashinfer_cutedsl`
- 如果配置错误，会触发 `AssertionError: forward_deepgemm_masked is deprecated`

#### 修复后（预期行为）
- 使用 `--moe-runner-backend auto`（默认）
- **应该自动选择 `flashinfer_cutedsl`**
- **应该看到日志**：`TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.`
- 服务器应该成功启动

#### 测试命令

```powershell
# 1. 应用修复
cd d:\Yc-software\github_code\sglang
git apply bug_16952_analysis\code\patch_server_args.py.patch
git apply bug_16952_analysis\code\patch_ep_moe_layer.py.patch

# 2. 启动容器并挂载代码
docker stop sglang_bug16952 2>$null
docker rm sglang_bug16952 2>$null
docker run -itd --gpus all --shm-size 32g -p 32123:32123 -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" -v "${PWD}:/sgl-workspace/sglang" --ipc=host --network=host --privileged --name sglang_bug16952 lmsysorg/sglang:latest /bin/bash

# 3. 重新安装 sglang
docker exec sglang_bug16952 bash -c "cd /sgl-workspace/sglang && pip install -e 'python[all]'"

# 4. 运行测试（修复后）
docker exec -it sglang_bug16952 python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat --tp 1 --ep 1 --port 32123 --trust-remote-code --enable-dp-attention --disable-custom-all-reduce --enable-two-batch-overlap --moe-a2a-backend deepep --deepep-mode normal --log-level info
```

#### 预期结果（修复后）
- ✅ 看到日志：`TBO/SBO is enabled with DeepEP. Automatically using flashinfer_cutedsl backend for overlap optimization.`
- ✅ 服务器成功启动
- ✅ 没有 `AssertionError` 错误

---

### 2. 验证错误检查逻辑（双重检查）

即使自动选择逻辑工作，我们也应该测试双重检查是否有效。

#### 测试场景：强制使用错误的配置

**注意**：这需要修改代码来强制使用错误的 backend，或者使用不支持 flashinfer_cutedsl 的环境。

#### 预期行为（修复后）
- 如果配置错误（TBO + DeepEP 但没有 flashinfer_cutedsl），应该看到：
  ```
  RuntimeError: TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. Please use --moe-runner-backend flashinfer_cutedsl.
  ```
- 而不是：`AssertionError: forward_deepgemm_masked is deprecated`

---

## ❌ 无法验证的内容（在 RTX 4090 上）

### 1. 复现原始 Bug
- **原因**：CUDA Graph 被禁用，不会进入 capture 阶段
- **需要**：支持 SM >= 90 的 GPU（H100/H20）

### 2. 验证 CUDA Graph capture 阶段的修复
- **原因**：同上
- **需要**：支持 SM >= 90 的 GPU

---

## 🔍 代码审查验证

即使无法在 RTX 4090 上复现 Bug，我们仍然可以通过代码审查验证修复：

### 1. 修复逻辑检查

#### server_args.py 修复
- ✅ 检查 TBO/SBO 是否启用
- ✅ 检查 DeepEP 是否使用
- ✅ 自动选择 `flashinfer_cutedsl`
- ✅ 如果不可用，抛出清晰的错误

#### layer.py 修复
- ✅ 在 fallback 到 `forward_deepgemm_masked` 之前检查
- ✅ 如果 TBO/SBO 启用但没有 `flashinfer_cutedsl`，抛出清晰的错误
- ✅ 错误信息清晰，符合 SGLang 的设计风格

### 2. 覆盖范围检查

- ✅ 覆盖了 TBO（`enable_two_batch_overlap`）
- ✅ 覆盖了 SBO（`enable_single_batch_overlap`）
- ✅ 覆盖了 `down_gemm_overlap_args` 参数
- ✅ 双重检查确保即使自动选择失败也能捕获错误

---

## 📋 完整验证清单

### 在 RTX 4090 上可以验证

- [ ] **自动选择逻辑**：修复后应该看到自动选择日志
- [ ] **服务器启动**：修复后服务器应该成功启动
- [ ] **代码审查**：修复逻辑正确，覆盖所有情况
- [ ] **错误信息**：如果配置错误，错误信息清晰

### 需要 SM >= 90 GPU 才能验证

- [ ] **复现原始 Bug**：在修复前应该看到 `AssertionError: forward_deepgemm_masked is deprecated`
- [ ] **验证修复**：修复后应该看到自动选择日志，服务器成功启动
- [ ] **CUDA Graph capture**：验证在 capture 阶段修复是否有效

---

## 🎯 建议的验证流程

### 阶段 1：在 RTX 4090 上验证（当前）

1. ✅ 应用修复代码
2. ✅ 测试自动选择逻辑（查看日志）
3. ✅ 验证服务器能否启动
4. ✅ 代码审查

### 阶段 2：在支持 SM >= 90 的 GPU 上验证（如果有访问权限）

1. 复现原始 Bug（修复前）
2. 应用修复代码
3. 验证修复是否有效（修复后）
4. 验证 CUDA Graph capture 阶段

---

## 📝 验证结果记录

在 `bug_16952_analysis/result/SUMMARY.md` 中记录：

```markdown
## 验证结果

### RTX 4090 验证（CUDA Graph 禁用）
- [ ] 自动选择逻辑：✅/❌
- [ ] 服务器启动：✅/❌
- [ ] 日志信息：✅/❌

### SM >= 90 GPU 验证（CUDA Graph 启用）
- [ ] 复现原始 Bug：✅/❌
- [ ] 验证修复：✅/❌
- [ ] CUDA Graph capture：✅/❌
```

---

## 💡 总结

**虽然无法在 RTX 4090 上复现原始 Bug，但我们可以：**

1. ✅ **验证修复逻辑**：自动选择逻辑是否正确
2. ✅ **验证错误处理**：双重检查是否有效
3. ✅ **代码审查**：修复是否完整、正确
4. ✅ **部分验证**：服务器能否正常启动

**要完全验证修复，需要：**
- 在支持 SM >= 90 的 GPU 上测试（H100/H20）
- 或者依赖代码审查和逻辑验证
