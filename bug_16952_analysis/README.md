# Bug #16952 分析文档

## 快速开始 ⭐

**新用户**: 查看 [QUICK_START.md](./QUICK_START.md) - 5 分钟快速开始

**完整流程**: 查看 [WORKFLOW.md](./WORKFLOW.md) - 从 Docker 到测试完成的详细步骤

---

## 快速总结

**问题**: 使用 `--enable-two-batch-overlap` + `--moe-a2a-backend deepep` 时报错

**原因**: 代码回退到已弃用的 `forward_deepgemm_masked`，该函数不支持 TBO/SBO

**修复**: 
- 自动选择 `flashinfer_cutedsl` backend（在 `server_args.py`）
- 双重检查，如果配置错误，抛出清晰的错误信息（在 `ep_moe/layer.py`）

**详情**: 见 [FIX_PLAN.md](./FIX_PLAN.md)

**测试**: 
- 完整流程: 见 [WORKFLOW.md](./WORKFLOW.md) ⭐ **推荐从这里开始**
- Docker 指南: 见 [DOCKER_TESTING_GUIDE.md](./DOCKER_TESTING_GUIDE.md)
- **验证计划**: 见 [VALIDATION_PLAN.md](./VALIDATION_PLAN.md) ⚠️ **重要：RTX 4090 限制说明**

## 文档索引

### 核心文档

1. **[FIX_PLAN.md](./FIX_PLAN.md)** ⭐ **主要文档**
   - 问题描述和用户需求
   - 根本原因分析
   - 修复方案和代码修改
   - 技术细节和影响分析
   - 测试计划

2. **[DOCKER_TESTING_GUIDE.md](./DOCKER_TESTING_GUIDE.md)** ⭐ **测试指南**
   - 为什么必须使用 Docker
   - Docker 测试步骤
   - RTX 4090 测试说明
   - 复现 Bug 的步骤

3. **[user_requirement.md](./user_requirement.md)**
   - 用户需求深度分析
   - 用户意图理解

4. **[MODEL_DOWNLOAD_EXPLAINED.md](./MODEL_DOWNLOAD_EXPLAINED.md)**
   - 模型下载说明
   - Docker vs 本地环境

5. **[original_issue.md](./original_issue.md)**
   - GitHub issue 的原始内容
   - 错误堆栈、复现步骤、环境信息

### 测试相关

- **[result/TEST_CHECKLIST.md](./result/TEST_CHECKLIST.md)** ⭐ **PR 测试清单**（专业测试矩阵，可直接贴到 PR）
- **[result/SUMMARY.md](./result/SUMMARY.md)** ⭐ **测试结果摘要**（环境信息 + 最小可复现命令 + 修复前后对照）
- **[VALIDATION_PLAN.md](./VALIDATION_PLAN.md)** ⚠️ **验证计划**（重要：RTX 4090 限制说明）
- **[test/COMMANDS.md](./test/COMMANDS.md)** ⭐ **直接复制运行的命令**（推荐，最简单）
- **[test/README_DEBUG.md](./test/README_DEBUG.md)** - 测试脚本说明
- **[test/docker_test.ps1](./test/docker_test.ps1)** - Docker 测试脚本（可选）

### 代码相关

- **[code/README.md](./code/README.md)** - 代码修复说明和补丁

### 工作流程

- **[WORKFLOW.md](./WORKFLOW.md)** ⭐ **完整测试流程**
  - 从 Docker 环境准备到测试完成的详细步骤
  - 每一步都有详细说明和命令

## 状态

- [x] 问题分析完成
- [x] 修复计划制定完成
- [x] 全局影响分析完成
- [x] 测试计划完善（包括向后兼容性测试）
- [x] 代码修复已应用
- [ ] 测试验证（重点：向后兼容性）
- [ ] 提交 PR

### 修复代码已应用

修复代码已经直接应用到以下文件：
1. **`python/sglang/srt/server_args.py`** - 自动选择 `flashinfer_cutedsl`
2. **`python/sglang/srt/layers/moe/ep_moe/layer.py`** - 双重检查逻辑

详见：[FIX_PLAN.md](./FIX_PLAN.md)

## 相关文件

- `python/sglang/srt/layers/moe/ep_moe/layer.py` - 主要修复文件
- `python/sglang/srt/server_args.py` - 自动选择逻辑
- `python/sglang/srt/single_batch_overlap.py` - SBO 相关代码
- `python/sglang/srt/layers/moe/utils.py` - `is_tbo_enabled()` 函数

## 关键注意事项

⚠️ **测试重点**:
1. **向后兼容性测试**是最重要的 - 确保不使用 TBO/SBO 的用户不受影响
2. **全局影响测试** - 确保修改不会影响其他使用场景
3. **不同模型测试** - Qwen3-MoE, DeepSeek V2 等
4. **回归测试** - 运行现有测试套件，确保没有破坏现有功能

⚠️ **强烈建议使用 Docker 进行测试**:
- DeepEP 安装复杂，Docker 镜像已预装
- NCCL 版本正确，避免通信错误
- 复现一致性，与用户环境一致

详见：[DOCKER_TESTING_GUIDE.md](./DOCKER_TESTING_GUIDE.md)
