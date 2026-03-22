# A01_B04：验证与测试（修复能力边界）

**Issue**: https://github.com/sgl-project/sglang/issues/20953  

本页说明：**如何证明「metadata 修复」有效**，以及 **哪些问题不在本次修复范围内**。

---

## 1. Issue 中的三层现象（对照 #20953）

| 代号 | 现象 | 本次 `rename_wheels*` 修复是否覆盖 |
|------|------|-----------------------------------|
| **A** | `pip install … --index-url …/cu130/` → `inconsistent version`（文件名 `+cu130`，`METADATA` 仍为 `0.4.0`） | **是**（发布后新 wheel 应对齐） |
| **B** | `ImportError: libnvrtc.so.12`（`.so` 已存在但加载失败） | **否**（链接/镜像/CUDA 13 NVRTC，需另跟） |
| **C** | `sglang` 要求 `cuda-python==12.9`，镜像为 `13.x` | **否**（主包依赖矩阵 / 镜像） |

**结论**：验证时要 **分开写结果**，避免「A 过了就说 #20953 全好了」。

---

## 2. 自动化冒烟（无 GPU，只测 **A** 的脚本逻辑）

### 2.1 CUDA：`rename_wheels.sh`

**脚本**：[`code/test_rename_wheels_smoke.sh`](../code/test_rename_wheels_smoke.sh)  

**环境**：Linux / **Git Bash**；`python` 能 `import wheel`；`unzip` 在 PATH。  
Windows 上若 `python3` 指向 Microsoft Store 占位符，请用 **conda 的 `python`** 或把 conda 放在 PATH 前面。

```bash
# 仓库根目录
bash bug_20953_analysis/code/test_rename_wheels_smoke.sh
```

**断言**：

- 输出 wheel 的 **`METADATA`** 含 `Version: 0.4.0+cu130`
- **`WHEEL`** 中 **`Tag`** 含 `manylinux2014_x86_64`（不是 `linux_x86_64`）

**说明**：使用 `SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130`，**不依赖** `/usr/local/cuda-*` 目录布局。

### 2.2 ROCm：`rename_wheels_rocm.sh`（可选）

**脚本**：[`code/test_rename_wheels_rocm_smoke.sh`](../code/test_rename_wheels_rocm_smoke.sh)  

```bash
bash bug_20953_analysis/code/test_rename_wheels_rocm_smoke.sh
```

使用 `SGL_KERNEL_ROCM_SUFFIX_OVERRIDE`（例如 `+rocm702`），逻辑与 CUDA 冒烟一致；拷贝的是 **`3rdparty/amd/wheel/sgl-kernel/rename_wheels_rocm.sh`**（与当前 upstream 路径一致）。

---

## 3. 手工检查任意已下载的 `.whl`（仍只验证 **A**）

**脚本**：[`code/verify_wheel_metadata.sh`](../code/verify_wheel_metadata.sh)

```bash
bash bug_20953_analysis/code/verify_wheel_metadata.sh path/to/sgl_kernel-0.4.0+cu130-....whl
```

会打印：

- `METADATA` 的 `Version:` 行  
- `WHEEL` 的 `Tag:` 行  
- 文件名是否含 `+cu` / `+rocm` 与 `Version` 是否 **同时** 含相同 local version（简单字符串检查）

---

## 4. 真机 / 镜像验证（接近用户场景）

### 4.1 验证 **A**（metadata + index）

在 **CUDA 13** 或任意能联网的环境：

```bash
pip install sglang-kernel --index-url https://docs.sglang.ai/whl/cu130/ --dry-run
# 或实际安装（需接受环境与依赖）
```

**通过标准**：**不再**出现 `inconsistent version: expected '…+cu130', but metadata has '0.4.0'`。  
**前提**：须使用 **修复后流水线新发布** 的 wheel；旧 release 上的文件不会自动变。

### 4.2 验证 **B**（NVRTC）

在 **`lmsysorg/sglang:v0.5.9-cu130`**（或同类）中，安装 kernel 后：

```bash
python -c "import sgl_kernel"
```

**若仍报 `libnvrtc.so.12`**：说明 **B 未解决**，需在 issue 中单开跟踪（构建链 / rpath / 镜像库路径）。

### 4.3 验证 **C**（cuda-python）

```bash
pip check
# 或 pip install 时的 dependency conflict 提示
```

与 **kernel 重打包** 无直接关系。

---

## 5. CI / 发布侧（维护者视角）

| 场景 | 建议 |
|------|------|
| PR 改 `sgl-kernel/rename_wheels.sh` | 跑 **`sgl-kernel-build-wheels`**（当前 matrix 多为 cu12.9，可能 **不带** `+cu*`，不覆盖 cu130 后缀路径）；合并前可用本文 **§2 冒烟** 补洞。 |
| Release `release-whl-kernel.yml` 打 **cu130** | 产物应用 **§3** 或 **§4.1** 抽检验证。 |
| 将来 PR 启用 **cuda 13.0** matrix | 注意 `scripts/ci/cuda/ci_install_dependency.sh` 里 **写死 wheel 文件名** 是否与 `+cu130` 文件名一致（建议改为 `sgl-kernel/dist/*.whl` 或 glob）。 |

---

## 6. 相关文档

- [A01_B00 超白话说明](./A01_B00_超白话说明.md)  
- [A01_B03 修复设计](./A01_B03_fix_design.md)  

---

*最后更新：与 upstream `3rdparty/amd/wheel/sgl-kernel/` ROCm 路径对齐。*
