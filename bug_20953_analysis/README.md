# Bug #20953 分析与验证

**Issue**: https://github.com/sgl-project/sglang/issues/20953  

## 核心文档（验证 / 测试）

| 文档或脚本 | 说明 |
|------------|------|
| [analysis/diagram_bug_and_fix.md](./analysis/diagram_bug_and_fix.md) | **图解**：#20953 bug 是什么、只 `mv` 为何不够、两种修对路径（Mermaid） |
| [analysis/A01_B04_verification_and_tests.md](./analysis/A01_B04_verification_and_tests.md) | **验证与测试**：分层（metadata / NVRTC / cuda-python）、冒烟、手工与发布检查 |
| [code/test_rename_wheels_smoke.sh](./code/test_rename_wheels_smoke.sh) | **CUDA** `rename_wheels.sh` 冒烟（`SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130`） |
| [code/test_rename_wheels_rocm_smoke.sh](./code/test_rename_wheels_rocm_smoke.sh) | **ROCm** `rename_wheels_rocm.sh` 冒烟（默认 `+rocm702`，可改 env） |
| [code/verify_wheel_metadata.sh](./code/verify_wheel_metadata.sh) | 对**任意** `.whl` 快速核对 `METADATA`/`WHEEL` 与文件名是否一致 |

## 上游代码路径（当前 main）

- CUDA：`sgl-kernel/rename_wheels.sh`（`sgl-kernel/Dockerfile` 构建末尾调用）
- ROCm：`3rdparty/amd/wheel/sgl-kernel/rename_wheels_rocm.sh`、`build_rocm.sh`（`release-whl-kernel.yml` 会 `cp …/wheel/sgl-kernel/*` 到 `sgl-kernel/`）

## Issue 一句话（Layer A）

Wheel **文件名**含 local version（`+cu130` / `+rocm*`）时，zip 内 **`METADATA` 的 `Version:`** 必须一致，否则 `pip` 从 index 安装会报 *inconsistent version*。**NVRTC / cuda-python** 为另层问题，见 A01_B04。

## 快速跑冒烟（仓库根目录）

```bash
bash bug_20953_analysis/code/test_rename_wheels_smoke.sh
bash bug_20953_analysis/code/test_rename_wheels_rocm_smoke.sh
```

需：**bash**、**Python + `wheel`**、**unzip**；Windows 建议 **Git Bash** 且 PATH 优先 **conda `python`**（避免 Store `python3` 占位符）。
