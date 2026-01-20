# Bug #16952 分析文档

## 📋 文档结构

### A01: flashinfer_cutedsl 详解
- **[A01_flashinfer.md](./A01_flashinfer.md)** ⭐ **主文档**
  - flashinfer_cutedsl 基本概念
  - 关键特性（支持 FP4、支持 TBO、使用 CuteDSL）
  - 与其他 Backend 的区别
  - 为什么 TBO 需要 flashinfer_cutedsl

### A01_Bxx: 平行文档（与 A01 同级）

- **[A01_B01_fix_comparison.md](./A01_B01_fix_comparison.md)** ⭐ **修复前后代码对比**
  - 修复位置1：ServerArgs 层面
  - 修复位置2：Layer 层面
  - 修复理由与逻辑
  - 完整执行流程对比
  - 为什么这样修复

- **[A01_B02_original_solutions.md](./A01_B02_original_solutions.md)** ⭐ **原始解决方案汇总**
  - 用户原始解决方案（Nickcp39）：代码层面自动选择 backend
  - Fridge003 的解决方案：使用 fp8 checkpoint
  - 两种方案的关系和对比

- **[A01_B03_original_issue.md](./A01_B03_original_issue.md)** - **原始 Issue 内容**
  - Issue 链接和标题
  - 问题描述和错误堆栈
  - 复现步骤和环境信息

- **[A01_B04_recent_issues_analysis.md](./A01_B04_recent_issues_analysis.md)** ⭐ **近期 Issue 分析**
  - 最近一个月 SGLang issue 统计
  - MoE + CUDA Graph 高频故障确认
  - 核心故障模式分析

- **[A01_B05_learning_without_gpu.md](./A01_B05_learning_without_gpu.md)** ⭐ **无 GPU 学习指南**
  - 不需要 GPU 的学习方法
  - 代码层面分析
  - 模拟和理解方法
  - 学习路径建议

- **[A01_B04_recent_issues_analysis.md](./A01_B04_recent_issues_analysis.md)** ⭐ **近期 Issue 分析**
  - 最近一个月 SGLang issue 统计
  - MoE + CUDA Graph 高频故障确认
  - 核心故障模式分析
  - 验证你的观察：MoE 和 Graph 确实很火

---

## 🎯 快速导航

1. **想了解 flashinfer_cutedsl 是什么** → [A01_flashinfer.md](./A01_flashinfer.md) ⭐ **从这里开始**
2. **想了解修复前后对比** → [A01_B01_fix_comparison.md](./A01_B01_fix_comparison.md)
3. **想了解原始解决方案** → [A01_B02_original_solutions.md](./A01_B02_original_solutions.md)
4. **想查看原始 Issue** → [A01_B03_original_issue.md](./A01_B03_original_issue.md)
5. **想验证 MoE + Graph 是否真的很火** → [A01_B04_recent_issues_analysis.md](./A01_B04_recent_issues_analysis.md) ⭐ **最新分析**
6. **没有 GPU 怎么学习？** → [A01_B05_learning_without_gpu.md](./A01_B05_learning_without_gpu.md) ⭐ **无 GPU 学习指南**

---

## 📁 其他文件

### 测试相关
- `test/` - 测试脚本
- `result/` - 测试结果

### 代码相关
- `code/` - 修复代码和补丁

### 脚本
- `check_model_and_test.ps1` - 模型检查和测试脚本
- `run_test.ps1` - 运行测试脚本

---

## 📝 文档命名规则

- **A01_xxx.md**: 主文档（A01 系列）
- **A01_A01_xxx.md**: A01 的子文档（A01_A01 系列）
- **A01_B01_xxx.md**: A01 的平行文档（A01_B01 系列）
