# Fix for Bug #16952: Add double-check in moe_impl to prevent using deprecated forward_deepgemm_masked
# File: python/sglang/srt/layers/moe/ep_moe/layer.py
# Location: In moe_impl() method, around line 489-495

"""
Fix implementation for ep_moe/layer.py

This is a double-check to ensure that even if the auto-selection in server_args.py
fails or is bypassed, we still catch the error early with a clear message.

The fix adds a check before falling back to forward_deepgemm_masked.
"""

# Import needed at the top of ep_moe/layer.py (if not already present):
# from sglang.srt.layers.moe.utils import is_tbo_enabled, get_moe_runner_backend

# Code to modify in moe_impl() method:

def moe_impl_with_fix(
    self,
    dispatch_output: Union[TopKOutput, DeepEPLLOutput],
    down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None,
):
    """
    Original moe_impl with fix for Bug #16952.
    
    Modify the elif block around line 489-495.
    """
    # ... existing code ...
    
    elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        if get_moe_runner_backend().is_flashinfer_cutedsl():
            return self.forward_flashinfer_cutedsl(
                dispatch_output, down_gemm_overlap_args=down_gemm_overlap_args
            )
        
        # Fix for Bug #16952: Double-check to prevent using deprecated forward_deepgemm_masked
        # when TBO/SBO is enabled
        if down_gemm_overlap_args is not None or is_tbo_enabled():
            raise RuntimeError(
                "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. "
                "Please use --moe-runner-backend flashinfer_cutedsl."
            )
        
        assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
        return self.forward_deepgemm_masked(dispatch_output)
    
    # ... existing code ...
