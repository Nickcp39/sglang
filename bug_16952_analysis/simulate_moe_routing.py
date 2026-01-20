#!/usr/bin/env python3
"""
模拟 MoE routing 的动态性
不需要 GPU，纯 Python 模拟
"""

import numpy as np
from typing import Dict, List


def simulate_moe_routing(num_tokens: int, num_experts: int, top_k: int = 2) -> Dict[int, List[int]]:
    """
    模拟 MoE routing：每个 token 选择 top_k 个 expert
    
    Args:
        num_tokens: token 数量
        num_experts: expert 数量
        top_k: 每个 token 选择的 expert 数量
    
    Returns:
        expert_token_counts: {expert_id: [token_indices]}
    """
    # 模拟 routing 逻辑（随机分数）
    np.random.seed(42)  # 可复现
    routing_scores = np.random.rand(num_tokens, num_experts)
    
    # 每个 token 选择 top_k 个 expert
    top_k_experts = np.argsort(routing_scores, axis=1)[:, -top_k:]
    
    # 统计每个 expert 分配到的 tokens
    expert_token_counts = {}
    for token_idx, experts in enumerate(top_k_experts):
        for expert in experts:
            if expert not in expert_token_counts:
                expert_token_counts[expert] = []
            expert_token_counts[expert].append(token_idx)
    
    return expert_token_counts


def demonstrate_dynamic_shape():
    """
    演示 MoE 的动态 shape 问题
    """
    print("=" * 60)
    print("MoE Routing Dynamic Shape Demo")
    print("=" * 60)
    
    num_experts = 8
    top_k = 2
    
    # 模拟不同 batch size
    for batch_size in [1, 4, 8, 16]:
        num_tokens = batch_size * 10
        expert_token_counts = simulate_moe_routing(num_tokens, num_experts, top_k)
        
        print(f"\n--- Batch Size: {batch_size} (Total Tokens: {num_tokens}) ---")
        
        # 展示每个 expert 的 token 数量（动态的）
        expert_counts = [len(tokens) for tokens in expert_token_counts.values()]
        print(f"Expert token counts: {expert_counts}")
        print(f"  Min: {min(expert_counts)}, Max: {max(expert_counts)}, "
              f"Std: {np.std(expert_counts):.2f}")
        
        # 展示为什么这是动态的
        if batch_size > 1:
            print(f"  [WARNING] Shape is dynamic: each expert processes different number of tokens")
            print(f"  [WARNING] This makes CUDA Graph capture difficult!")


def demonstrate_cuda_graph_conflict():
    """
    演示 MoE 与 CUDA Graph 的冲突
    """
    print("\n" + "=" * 60)
    print("MoE + CUDA Graph 冲突演示")
    print("=" * 60)
    
    print("\n--- Scenario 1: Static Shape (CUDA Graph Friendly) ---")
    print("  Batch Size: 8")
    print("  Shape: [8, 128, 4096]  (fixed)")
    print("  [OK] CUDA Graph can capture this")
    
    print("\n--- Scenario 2: MoE Routing (Dynamic Shape) ---")
    expert_token_counts = simulate_moe_routing(80, 8, top_k=2)
    expert_counts = [len(tokens) for tokens in expert_token_counts.values()]
    print(f"  Expert token counts: {expert_counts}")
    print(f"  [WARNING] Shape is dynamic: {min(expert_counts)} to {max(expert_counts)}")
    print(f"  [ERROR] CUDA Graph cannot easily capture this")
    print(f"  [INSIGHT] This is why MoE + CUDA Graph is problematic!")


def demonstrate_tbo_requirement():
    """
    演示为什么 TBO 需要 flashinfer_cutedsl
    """
    print("\n" + "=" * 60)
    print("TBO 为什么需要 flashinfer_cutedsl")
    print("=" * 60)
    
    print("\n--- Backend Comparison ---")
    
    print("\n1. forward_deepgemm_masked (does NOT support TBO)")
    print("   Signature: forward_deepgemm_masked(dispatch_output)")
    print("   [ERROR] No down_gemm_overlap_args parameter")
    print("   [ERROR] Cannot support TBO overlap")
    
    print("\n2. forward_flashinfer_cutedsl (supports TBO)")
    print("   Signature: forward_flashinfer_cutedsl(dispatch_output, down_gemm_overlap_args)")
    print("   [OK] Has down_gemm_overlap_args parameter")
    print("   [OK] Can support TBO overlap")
    
    print("\n--- Why Your Fix is Important ---")
    print("  When TBO is enabled:")
    print("  - If backend is 'auto' -> may select forward_deepgemm_masked")
    print("  - forward_deepgemm_masked doesn't support TBO -> Bug!")
    print("  - Your fix: auto-select flashinfer_cutedsl when TBO + DeepEP")
    print("  - Result: Always use the correct backend [OK]")


if __name__ == "__main__":
    # 演示 1: MoE routing 的动态性
    demonstrate_dynamic_shape()
    
    # 演示 2: CUDA Graph 冲突
    demonstrate_cuda_graph_conflict()
    
    # 演示 3: TBO 要求
    demonstrate_tbo_requirement()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
1. MoE routing 导致动态 shape
   → 每个 batch 的 expert 分配不同
   → 内存访问模式不固定

2. CUDA Graph 要求静态执行
   → 所有内存地址必须固定
   → 算子执行流必须可预测

3. 冲突：动态 vs 静态
   → MoE + CUDA Graph 容易失败
   → 需要特殊处理（如你的修复）

4. TBO 需要特殊 backend
   → forward_deepgemm_masked 不支持
   → 必须使用 flashinfer_cutedsl
    """)
