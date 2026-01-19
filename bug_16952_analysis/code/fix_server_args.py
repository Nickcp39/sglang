# Fix for Bug #16952: Auto-select flashinfer_cutedsl backend when TBO/SBO + DeepEP is enabled
# File: python/sglang/srt/server_args.py
# Location: In _handle_model_specific_adjustments() method

"""
Fix implementation for server_args.py

Add this code block in the _handle_model_specific_adjustments() method,
after the existing model-specific adjustments (around line 797, before the method ends).

The fix automatically selects flashinfer_cutedsl backend when:
- TBO or SBO is enabled
- DeepEP (moe_a2a_backend == "deepep") is used
- moe_runner_backend is "auto" or None
"""

# Import needed at the top of server_args.py (if not already present):
# from sglang.srt.layers.moe.utils import get_moe_runner_backend

# Code to add in _handle_model_specific_adjustments():

def _handle_model_specific_adjustments_with_fix(self):
    """
    Original method with fix for Bug #16952.
    
    Add this block at the end of _handle_model_specific_adjustments(),
    before the method returns.
    """
    # ... existing code ...
    
    # Fix for Bug #16952: Auto-select flashinfer_cutedsl when TBO/SBO + DeepEP is enabled
    if self.enable_two_batch_overlap or self.enable_single_batch_overlap:
        if self.moe_a2a_backend == "deepep":
            if self.moe_runner_backend == "auto" or self.moe_runner_backend is None:
                # Check if flashinfer_cutedsl is available
                try:
                    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
                    self.moe_runner_backend = "flashinfer_cutedsl"
                    logger.info(
                        "TBO/SBO is enabled with DeepEP. "
                        "Automatically using flashinfer_cutedsl backend for overlap optimization."
                    )
                except ImportError:
                    raise RuntimeError(
                        "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl, "
                        "but flashinfer_cutedsl is not available. "
                        "Please install flashinfer with cutedsl support."
                    )
            elif self.moe_runner_backend != "flashinfer_cutedsl":
                raise RuntimeError(
                    "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl. "
                    "Please use --moe-runner-backend flashinfer_cutedsl."
                )
