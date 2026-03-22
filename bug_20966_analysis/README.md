# Bug #20966 分析与测试设计

**Issue**: https://github.com/sgl-project/sglang/issues/20966  

本目录仅做**问题分析**与**测试方案设计**。按当前要求：**不执行**任何脚本/测试，**不修改**仓库内产品代码（分析文档除外）。

## 文档索引

| 路径 | 说明 |
|------|------|
| [analysis/design_and_tests.md](./analysis/design_and_tests.md) | 根因与设计要点、验证分层、建议测试用例（含 pytest / 手工 / GPU 冒烟） |

## Issue 一句话

在 **未指定 `--speculative-draft-model-path`** 时，EAGLE 草稿 worker 会回退到与 target **同一** checkpoint，对 MiniMax M2 会再建一份完整 MoE 模型，极易在加载草稿权重阶段 **CUDA OOM**；与 `--mem-fraction-static` 调参无本质关系。

## 说明

- 协作方建议的 **Aurora-Spec** 草稿仓库，属于**正确用法**（独立 EAGLE 草稿权重），见主分析文档。
- 若后续在本仓库增加 **启动期校验** 或 **文档**，验证方式以 `analysis/design_and_tests.md` 中的分层测试为准。
