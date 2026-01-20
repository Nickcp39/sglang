#!/usr/bin/env python3
"""
模拟 CUDA Graph capture 的限制
不需要 GPU，纯 Python 模拟
"""

from typing import List, Tuple, Dict, Optional


class SimulatedCudaGraph:
    """
    模拟 CUDA Graph capture 的限制
    """
    def __init__(self):
        self.captured_ops: List[Tuple[str, Dict]] = []
        self.capture_mode = False
        self.capture_failed = False
        self.failure_reason = None
    
    def start_capture(self):
        """开始 capture"""
        self.capture_mode = True
        self.captured_ops = []
        self.capture_failed = False
        self.failure_reason = None
        print("[OK] CUDA Graph capture started")
    
    def end_capture(self) -> bool:
        """结束 capture"""
        self.capture_mode = False
        if self.capture_failed:
            print(f"[ERROR] CUDA Graph capture failed: {self.failure_reason}")
            return False
        else:
            print(f"[OK] CUDA Graph captured {len(self.captured_ops)} operations")
            return True
    
    def execute_op(self, op_name: str, **kwargs) -> bool:
        """
        执行操作
        
        Returns:
            True if successful, False if failed
        """
        if self.capture_mode:
            # 检查是否允许 capture
            if self._check_capture_allowed(op_name, **kwargs):
                self.captured_ops.append((op_name, kwargs))
                print(f"  [OK] Captured: {op_name}")
                return True
            else:
                self.capture_failed = True
                self.failure_reason = self._get_reason(op_name, **kwargs)
                print(f"  [ERROR] Cannot capture {op_name}: {self.failure_reason}")
                return False
        else:
            print(f"  [OK] Executed: {op_name}")
            return True
    
    def _check_capture_allowed(self, op_name: str, **kwargs) -> bool:
        """检查操作是否允许 capture"""
        # CUDA Graph 不允许的操作
        forbidden_ops = [
            "dynamic_memory_alloc",
            "random_number_generation",
            "nccl_all_to_all",  # 某些情况下
            "conditional_branch",  # 某些情况下
        ]
        
        if op_name in forbidden_ops:
            return False
        
        # 检查动态 shape
        if "shape" in kwargs:
            if kwargs.get("is_dynamic", False):
                return False
        
        return True
    
    def _get_reason(self, op_name: str, **kwargs) -> str:
        """获取不允许 capture 的原因"""
        reasons = {
            "dynamic_memory_alloc": "Dynamic memory allocation not allowed during capture",
            "random_number_generation": "RNG not allowed during capture",
            "nccl_all_to_all": "NCCL communication may not be allowed during capture",
            "conditional_branch": "Conditional branches may not be allowed during capture",
        }
        
        if "shape" in kwargs and kwargs.get("is_dynamic", False):
            return "Dynamic shape not allowed during capture"
        
        return reasons.get(op_name, "Unknown reason")


def simulate_normal_forward():
    """
    模拟正常的 forward（可以 capture）
    """
    print("=" * 60)
    print("Scenario 1: Normal Forward (CUDA Graph Friendly)")
    print("=" * 60)
    
    graph = SimulatedCudaGraph()
    graph.start_capture()
    
    # 正常的操作（静态 shape）
    graph.execute_op("matmul", shape=(8, 128, 4096), is_dynamic=False)
    graph.execute_op("add", shape=(8, 128, 4096), is_dynamic=False)
    graph.execute_op("layer_norm", shape=(8, 128, 4096), is_dynamic=False)
    
    success = graph.end_capture()
    return success


def simulate_moe_forward():
    """
    模拟 MoE forward（难以 capture）
    """
    print("\n" + "=" * 60)
    print("Scenario 2: MoE Forward (CUDA Graph Problematic)")
    print("=" * 60)
    
    graph = SimulatedCudaGraph()
    graph.start_capture()
    
    # 正常的操作
    graph.execute_op("attention", shape=(8, 128, 4096), is_dynamic=False)
    
    # MoE routing（动态 shape）
    print("\n  --- MoE Routing (Dynamic) ---")
    success = graph.execute_op("moe_routing", shape=(80, 8), is_dynamic=True)
    
    if not success:
        print("\n  [INSIGHT] This is why MoE + CUDA Graph fails!")
        print("  [INSIGHT] MoE routing creates dynamic shapes")
        print("  [INSIGHT] CUDA Graph requires static shapes")
    
    graph.end_capture()
    return success


def simulate_moe_with_nccl():
    """
    模拟 MoE + NCCL All-to-All（难以 capture）
    """
    print("\n" + "=" * 60)
    print("Scenario 3: MoE + NCCL All-to-All (CUDA Graph Problematic)")
    print("=" * 60)
    
    graph = SimulatedCudaGraph()
    graph.start_capture()
    
    # MoE routing
    graph.execute_op("moe_routing", shape=(80, 8), is_dynamic=True)
    
    # NCCL All-to-All（Expert Parallelism）
    print("\n  --- NCCL All-to-All (Expert Parallelism) ---")
    success = graph.execute_op("nccl_all_to_all", num_tokens=80)
    
    if not success:
        print("\n  [INSIGHT] This is why EP (Expert Parallel) + CUDA Graph fails!")
        print("  [INSIGHT] NCCL communication may not be allowed during capture")
    
    graph.end_capture()
    return success


def simulate_tbo_overlap():
    """
    模拟 TBO Overlap（需要特殊 backend）
    """
    print("\n" + "=" * 60)
    print("Scenario 4: TBO Overlap (Requires Special Backend)")
    print("=" * 60)
    
    print("\n  --- TBO Overlap Requirements ---")
    print("  TBO needs:")
    print("  1. Multi-stream execution")
    print("  2. Overlap parameters (down_gemm_overlap_args)")
    print("  3. Special backend support (flashinfer_cutedsl)")
    
    print("\n  --- Backend Comparison ---")
    print("  forward_deepgemm_masked:")
    print("    [ERROR] No down_gemm_overlap_args parameter")
    print("    [ERROR] Cannot support TBO")
    
    print("\n  forward_flashinfer_cutedsl:")
    print("    [OK] Has down_gemm_overlap_args parameter")
    print("    [OK] Can support TBO")
    
    print("\n  [INSIGHT] This is why your fix is important!")
    print("  [INSIGHT] Auto-select flashinfer_cutedsl when TBO + DeepEP")


def demonstrate_why_fix_is_needed():
    """
    演示为什么需要你的修复
    """
    print("\n" + "=" * 60)
    print("Why Your Fix is Needed")
    print("=" * 60)
    
    print("""
Problem:
  - User enables TBO + DeepEP
  - moe_runner_backend is "auto"
  - Code may select forward_deepgemm_masked (doesn't support TBO)
  - CUDA Graph capture fails or deprecated assert triggers

Your Fix:
  1. ServerArgs layer: Auto-select flashinfer_cutedsl when TBO + DeepEP
  2. Layer layer: Defensive check to prevent wrong backend

Result:
  - Always use correct backend (flashinfer_cutedsl)
  - TBO works correctly
  - CUDA Graph capture succeeds (if other conditions are met)
    """)


if __name__ == "__main__":
    # 场景 1: 正常 forward
    simulate_normal_forward()
    
    # 场景 2: MoE forward
    simulate_moe_forward()
    
    # 场景 3: MoE + NCCL
    simulate_moe_with_nccl()
    
    # 场景 4: TBO Overlap
    simulate_tbo_overlap()
    
    # 总结
    demonstrate_why_fix_is_needed()
    
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. CUDA Graph capture has strict requirements:
   - Static memory addresses
   - Predictable execution flow
   - No dynamic shapes (in most cases)
   - No NCCL communication (in some cases)

2. MoE introduces dynamic behavior:
   - Dynamic routing (different expert assignment per batch)
   - Dynamic shapes (different token counts per expert)
   - NCCL All-to-All communication (Expert Parallelism)

3. Conflict: Dynamic vs Static
   - MoE's dynamic nature conflicts with CUDA Graph's static requirements
   - This is why MoE + CUDA Graph is a "high-frequency failure zone"

4. Solutions:
   - Use special backends (flashinfer_cutedsl)
   - Add defensive checks (your fix)
   - Disable CUDA Graph for MoE (workaround)
   - Use piecewise CUDA Graph (experimental)
    """)
