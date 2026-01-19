# Issue #16952 原始内容

## Issue 链接
https://github.com/sgl-project/sglang/issues/16952

## 问题标题
[Bug] AssertionError: forward_deepgemm_masked is deprecated----Qwen3-30B #16952

## 提交者
@ziyuhuang123

## 问题描述

### 现象
- 不使用 TBO 和 DeepEP 时，Qwen30B 可以正常运行
- 使用 TBO 和 DeepEP 时，报错

### 错误堆栈
```
[2026-01-12 11:18:13] Received sigquit from a child process. It usually means the child failed.
[2026-01-12 11:18:13 TP1 EP1] Scheduler hit an exception: Traceback (most recent call last):
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 2905, in run_scheduler_process
    scheduler = Scheduler(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 333, in __init__
    self.init_model_worker()
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 471, in init_model_worker
    self.tp_worker = TpModelWorker(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/managers/tp_worker.py", line 253, in __init__
    self._model_runner = ModelRunner(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 355, in __init__
    self.initialize(min_per_gpu_memory)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 541, in initialize
    self.init_device_graphs()
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 1991, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 358, in __init__
    self.capture()
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 513, in capture
    _capture_one_stream()
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 500, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 719, in capture_one_batch_size
    run_once()
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 706, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/models/qwen3_moe.py", line 919, in forward
    hidden_states = self.model(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/models/qwen2_moe.py", line 638, in forward
    hidden_states, residual = layer(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/models/qwen3_moe.py", line 792, in forward
    hidden_states = self.mlp(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/models/qwen3_moe.py", line 277, in forward
    return self.forward_deepep(hidden_states, forward_batch)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/models/qwen3_moe.py", line 325, in forward_deepep
    final_hidden_states = self.experts(
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/layers/moe/ep_moe/layer.py", line 167, in forward
    return self.forward_impl(hidden_states, topk_output)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/layers/moe/ep_moe/layer.py", line 185, in forward_impl
    combine_input = self.run_moe_core(dispatch_output)
  File "/opt/conda/envs/new/lib/python3.10/site-packages/sglang/srt/layers/moe/ep_moe/layer.py", line 235, in run_moe_core
    assert False, "forward_deepgemm_masked is deprecated"
AssertionError: forward_deepgemm_masked is deprecated
```

### 复现步骤
```bash
python3 -m sglang.launch_server \
    --model-path /dev/shm/qwen30b/qwen30b \
    --tp 8 --ep 8 \
    --port 32123 \
    --trust-remote-code \
    --enable-dp-attention \
    --disable-custom-all-reduce \
    --enable-two-batch-overlap \
    --moe-a2a-backend deepep
```

### 环境信息
```
Python: 3.10.19 | packaged by conda-forge | (main, Oct 22 2025, 22:29:10) [GCC 14.3.0]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA H20
GPU 0,1,2,3,4,5,6,7 Compute Capability: 9.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.8, V12.8.61
CUDA Driver Version: 580.82.07
PyTorch: 2.9.1+cu128
sglang: 0.5.7
sgl_kernel: 0.3.20
flashinfer_python: 0.5.3
flashinfer_cubin: 0.5.3
flashinfer_jit_cache: Module Not Found
triton: 3.5.1
transformers: 4.57.1
torchao: 0.9.0
numpy: 2.2.6
aiohttp: 3.13.3
fastapi: 0.128.0
hf_transfer: 0.1.9
huggingface_hub: 0.36.0
interegular: 0.3.3
modelscope: 1.33.0
orjson: 3.11.5
outlines: 0.1.11
packaging: 25.0
psutil: 7.2.1
pydantic: 2.12.5
python-multipart: 0.0.21
pyzmq: 27.1.0
uvicorn: 0.40.0
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.1.27
openai: 2.6.1
tiktoken: 0.12.0
anthropic: 0.75.0
litellm: Module Not Found
decord2: 3.0.0
```

### Issue 状态
- **状态**: Open
- **认领者**: @Nickcp39 (commented 2 days ago)
- **认领信息**: "I'd like to take this issue. I can reproduce it on my setup. If I can't solve it within a few days, I'll update here."

### 关键信息
1. **错误位置**: `sglang/srt/layers/moe/ep_moe/layer.py`, line 235, in `run_moe_core`
2. **错误信息**: `assert False, "forward_deepgemm_masked is deprecated"`
3. **触发条件**: 
   - `--enable-two-batch-overlap` (TBO)
   - `--moe-a2a-backend deepep`
   - 运行 Qwen3-30B 模型
4. **正常情况**: 不使用 TBO 和 DeepEP 时可以正常运行
