#!/usr/bin/env python3
"""
Test script for Bug #16952 fix

This script tests both scenarios:
1. Without --moe-runner-backend (should auto-select flashinfer_cutedsl)
2. With explicit --moe-runner-backend flashinfer_cutedsl (should work)
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# Test configuration
MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
TP_SIZE = 2
EP_SIZE = 2
PORT = 32123
TIMEOUT = 300  # 5 minutes timeout

def run_test(test_name, args, should_succeed=True):
    """Run a test and check if it succeeds or fails as expected."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Command: python3 -m sglang.launch_server {' '.join(args)}")
    print()
    
    cmd = ["python3", "-m", "sglang.launch_server"] + args
    
    start_time = time.time()
    try:
        # Run with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if should_succeed:
            if result.returncode == 0:
                print(f"✅ Test PASSED (took {elapsed:.2f}s)")
                return True
            else:
                print(f"❌ Test FAILED (expected success, got return code {result.returncode})")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                return False
        else:
            if result.returncode != 0:
                # Check if it's the expected error
                if "TBO/SBO requires --moe-runner-backend flashinfer_cutedsl" in result.stderr or \
                   "AssertionError: forward_deepgemm_masked is deprecated" in result.stderr:
                    print(f"✅ Test PASSED (got expected error, took {elapsed:.2f}s)")
                    return True
                else:
                    print(f"⚠️  Test got error but not the expected one (took {elapsed:.2f}s)")
                    print(f"STDOUT:\n{result.stdout}")
                    print(f"STDERR:\n{result.stderr}")
                    return False
            else:
                print(f"❌ Test FAILED (expected error, but got success)")
                return False
    except subprocess.TimeoutExpired:
        print(f"⏱️  Test TIMEOUT (exceeded {TIMEOUT}s)")
        return False
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Bug #16952 Fix Test Suite")
    print("="*60)
    
    base_args = [
        "--model-path", MODEL_PATH,
        "--tp", str(TP_SIZE),
        "--ep", str(EP_SIZE),
        "--port", str(PORT),
        "--trust-remote-code",
        "--enable-dp-attention",
        "--disable-custom-all-reduce",
        "--enable-two-batch-overlap",
        "--moe-a2a-backend", "deepep",
    ]
    
    results = []
    
    # Test 1: Without --moe-runner-backend (should auto-select and work)
    test1_args = base_args.copy()
    results.append(("Test 1: Auto-select flashinfer_cutedsl", test1_args, True))
    
    # Test 2: With explicit --moe-runner-backend flashinfer_cutedsl (should work)
    test2_args = base_args + ["--moe-runner-backend", "flashinfer_cutedsl"]
    results.append(("Test 2: Explicit flashinfer_cutedsl", test2_args, True))
    
    # Test 3: With wrong backend (should fail with clear error)
    test3_args = base_args + ["--moe-runner-backend", "triton"]
    results.append(("Test 3: Wrong backend (should fail)", test3_args, False))
    
    # Run tests
    passed = 0
    total = len(results)
    
    for test_name, args, should_succeed in results:
        if run_test(test_name, args, should_succeed):
            passed += 1
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed}/{total} passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
