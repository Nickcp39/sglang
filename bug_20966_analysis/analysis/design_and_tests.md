# #20966：设计分析 + 测试方案（仅文档）

**Issue**: https://github.com/sgl-project/sglang/issues/20966  
**约束**：本文档阶段**不运行**测试命令、**不修改** `python/sglang/...` 等产品代码。

---

## 1. 现象与日志锚点

- **触发条件**：`--speculative-algorithm EAGLE`（及同类需要 draft worker 的路径），`--model-path` 指向 `MiniMaxAI/MiniMax-M2.5`，且**未**设置 `--speculative-draft-model-path`（或与 target 等价的路径）。
- **失败阶段**：`Scheduler.maybe_init_draft_worker` → `EAGLEWorker` / draft `TpModelWorker` → `load_model` → `MiniMaxM2ForCausalLM` → MoE FP8 `create_weights` → `torch.OutOfMemoryError`。
- **日志特征**：target 已 `Load weight begin` 且占用绝大部分显存后，draft 侧再次构建完整 `MiniMaxM2Model` 时 OOM。

---

## 2. 根因（设计层面）

### 2.1 Draft 路径回退

- Draft worker 在 `TpModelWorker._init_model_config` 中使用 `speculative_draft_model_path`；若为 `None`，`ModelConfig.from_server_args` 使用 `model_path or server_args.model_path`，即**与 target 相同的 `model_path`**。
- 因此未显式指定草稿仓库时，语义上等价于「再加载一份同名 checkpoint 作为 draft 模型」。

### 2.2 与 DeepSeek / MTP 类模型的差异

- 对部分架构，`ModelConfig._config_draft_model` 会把 draft 侧 **remap** 成 NextN / MTP 等小头，**同一目录**下只实例化小模型。
- **MiniMax M2**（`MiniMaxM2ForCausalLM`）当前**无**同类 remap；draft 与 target 同为完整 `MiniMaxM2Model`（`num_hidden_layers` 层 MoE 等）。
- **补充**：HF `config.json` 中可能存在 `use_mtp` / `num_mtp_modules` 等字段，但 SGLang 中 `MiniMaxM2Model` 的 `make_layers` 仅覆盖主 `num_hidden_layers`，**不**等同于「EAGLE 已自动使用 MTP 作为小草稿」。因此不能从「配置里有 MTP」推出「未指定 draft 路径也安全」。

### 2.3 与 `mem_fraction-static` 的关系

- OOM 发生在**第二套模型权重分配**阶段，显存已被第一套占满；调大/调小静态比例至多改变 KV/池化预留，**不能**把「两套完整 MiniMax」变成「一套」，故非根因。

### 2.4 预期正确用法（产品行为，非本文档实施范围）

- 使用 **独立 EAGLE 草稿**（例如协作方提到的 `togethercomputer/Aurora-Spec-Minimax-M2.5`），通过 `--speculative-draft-model-path`（及必要时 `--speculative-draft-model-revision`）指定；target `--model-path` 仍为基座 MiniMax。

---

## 3. 修复/改进的设计选项（记录用，本阶段不实现）

| 方向 | 思路 | 验证侧重点 |
|------|------|------------|
| **A. 启动期失败快** | 对 `MiniMaxM2ForCausalLM` + EAGLE（及 STANDALONE/EAGLE3 若同逻辑）且未设置 draft 路径时 `ValueError`，文案指向 Aurora-Spec 类仓库 | 无 GPU：参数校验单测；有 GPU：应在建 draft 前失败 |
| **B. 文档与 CLI help** | `launch_server` / 文档明确：MiniMax EAGLE 必须带 draft 路径 | 文档 review；可选 e2e 与 A 一致 |
| **C. 长期** | 若上游支持：同 checkpoint MTP 作为 EAGLE draft（需独立设计与大量实现） | 单测 + 多卡集成 + 精度/吞吐基准 |

---

## 4. 测试设计（分层，均仅规划）

### 4.1 层级定义

| 层级 | 目的 | 依赖 |
|------|------|------|
| **L0** | 确认代码路径与回退语义（静态阅读 / 单次 import 不启动 CUDA） | 源码 |
| **L1** | ServerArgs  speculative 分支：mock `get_model_config().hf_config.architectures` 为 `MiniMaxM2ForCausalLM`，无 draft 路径时期望抛错（若实现 A） | pytest，`unittest.mock` |
| **L2** | 最小 GPU：双卡或小 TP 若可加载单卡 MiniMax 仍过大则跳过；否则「带 Aurora draft 路径能过 init；不带则快速失败或 OOM」 | 真实权重 + GPU |
| **L3** | 完整 EAGLE 正确性、吞吐、长上下文 | 大集群，超出 #20966 范围 |

**说明**：L1 的实现细节需与 `ServerArgs._handle_speculative_decoding` / `check_server_args` 的调用顺序对齐；设计上是「在 draft worker 创建之前拦截」。

### 4.2 L1 建议用例（pytest 草图，不执行）

- **用例 1**：`speculative_algorithm="EAGLE"`，`speculative_draft_model_path=None`，`model_path` 任意合法字符串，patch `get_model_config` 返回 `architectures=["MiniMaxM2ForCausalLM"]` → 调用会触发 speculative 校验的入口 → **期望** `ValueError`，且消息含 `speculative-draft-model-path` 或 Aurora 示例（与最终实现一致即可）。
- **用例 2**：同上但 `speculative_draft_model_path="/path/to/aurora-spec"` → **期望** 不因 MiniMax 专用规则失败（除非路径解析失败，应区分）。
- **用例 3**（回归）：DeepSeek 类架构在「无 draft 路径」时仍走既有 MTP 逻辑，**不应**被 MiniMax 规则误伤（需在 patch 中使用 `DeepseekV3ForCausalLM` 等对照）。

### 4.3 L2 手工 / 冒烟清单（设计）

1. **负向（修复前或关闭 A 时）**：仅 issue 中命令，预期长时间加载后 **OOM** 或 **scheduler 异常**（与现网一致）。
2. **负向（修复 A 后）**：同样命令，预期 **数秒内** `ValueError` 清晰退出，无 GPU 泄漏（可 `nvidia-smi` 看进程）。
3. **正向**：增加 `--speculative-draft-model-path togethercomputer/Aurora-Spec-Minimax-M2.5`（及 trust / revision 等与官方一致），预期 **scheduler 完成初始化**；再发一条短 completion 确认服务存活（可选）。

### 4.4 非目标（避免误判）

- 不把「能跑 EAGLE」等同于「#20966 已解决」；必须覆盖 **无 draft 路径** 与 **有独立 draft** 两种配置。
- 不把 HF `num_mtp_modules` 与 SGLang EAGLE draft 自动关联，除非有对应实现与测试。

---

## 5. 关键源码锚点（便于 review）

- `python/sglang/srt/managers/tp_worker.py`：`is_draft_worker` 时 `model_path` / `model_revision` 来源。
- `python/sglang/srt/configs/model_config.py`：`from_server_args` 的 `model_path or server_args.model_path`；`_config_draft_model` 各架构 remap。
- `python/sglang/srt/managers/scheduler.py`：`maybe_init_draft_worker`。
- `python/sglang/srt/speculative/eagle_worker.py`：draft `TpModelWorker` 初始化链。
- `python/sglang/srt/models/minimax_m2.py`：`MiniMaxM2Model` 层数与 MoE 构建。
- `python/sglang/srt/server_args.py`：`_handle_speculative_decoding`、DeepSeek 块内对 `speculative_draft_model_path` 的赋值逻辑（与 MiniMax 对比）。

---

## 6. 与 #20953 文件夹的对应关系

- `bug_20953_analysis`：wheel metadata 等**发布物**验证，脚本可本地无 GPU 冒烟。
- `bug_20966_analysis`：以**运行时显存与 draft 加载路径**为主；L1 可完全无 GPU，L2 强依赖大模型与多卡环境。

本文档随 issue / PR 迭代时可增补「已实现用例路径」与「CI 是否收录」两列，便于追踪。
