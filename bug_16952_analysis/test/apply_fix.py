#!/usr/bin/env python3
"""
应用 Bug #16952 的修复代码

用法:
    python apply_fix.py

功能:
    1. 备份原始文件
    2. 应用修复到 server_args.py 和 ep_moe/layer.py
"""

import os
import shutil
from pathlib import Path

# 项目根目录（假设脚本在 bug_16952_analysis/test/ 目录下）
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 需要修改的文件
FILES_TO_FIX = {
    "python/sglang/srt/server_args.py": {
        "backup": "python/sglang/srt/server_args.py.backup",
        "fix_file": SCRIPT_DIR.parent / "code" / "fix_server_args.py",
    },
    "python/sglang/srt/layers/moe/ep_moe/layer.py": {
        "backup": "python/sglang/srt/layers/moe/ep_moe/layer.py.backup",
        "fix_file": SCRIPT_DIR.parent / "code" / "fix_ep_moe_layer.py",
    },
}

def backup_file(file_path):
    """备份文件"""
    backup_path = file_path + ".backup"
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份: {file_path} -> {backup_path}")
        return True
    else:
        print(f"❌ 文件不存在: {file_path}")
        return False

def apply_fix_from_code(file_path, fix_code_path):
    """从修复代码文件应用修复"""
    if not os.path.exists(fix_code_path):
        print(f"❌ 修复代码文件不存在: {fix_code_path}")
        return False
    
    # 读取修复代码（这里假设修复代码文件包含完整的修改说明）
    # 实际应用中，需要根据 fix_code_path 的内容来修改文件
    print(f"📝 应用修复: {file_path}")
    print(f"   修复代码: {fix_code_path}")
    
    # 这里应该根据实际的修复代码来修改文件
    # 由于修复代码可能包含上下文，需要智能匹配和替换
    print(f"⚠️  请手动查看修复代码并应用，或使用 Git patch 文件")
    return False

def check_fix_applied(file_path, pattern):
    """检查修复是否已应用"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return pattern in content

def main():
    print("=" * 60)
    print("Bug #16952 修复代码应用工具")
    print("=" * 60)
    print()
    
    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print()
    
    # 备份文件
    print("步骤 1: 备份原始文件")
    print("-" * 60)
    all_backed_up = True
    for file_path, config in FILES_TO_FIX.items():
        full_path = PROJECT_ROOT / file_path
        if backup_file(full_path):
            all_backed_up = True
        else:
            all_backed_up = False
    print()
    
    if not all_backed_up:
        print("❌ 备份失败，请检查文件路径")
        return 1
    
    # 检查修复是否已应用
    print("步骤 2: 检查修复状态")
    print("-" * 60)
    server_args_fixed = check_fix_applied(
        PROJECT_ROOT / "python/sglang/srt/server_args.py",
        "TBO/SBO is enabled with DeepEP"
    )
    layer_fixed = check_fix_applied(
        PROJECT_ROOT / "python/sglang/srt/layers/moe/ep_moe/layer.py",
        "TBO/SBO requires --moe-runner-backend"
    )
    
    if server_args_fixed:
        print("✅ server_args.py 修复已应用")
    else:
        print("❌ server_args.py 修复未应用")
    
    if layer_fixed:
        print("✅ ep_moe/layer.py 修复已应用")
    else:
        print("❌ ep_moe/layer.py 修复未应用")
    print()
    
    if server_args_fixed and layer_fixed:
        print("✅ 所有修复已应用！")
        return 0
    
    # 应用修复
    print("步骤 3: 应用修复")
    print("-" * 60)
    print("⚠️  自动应用功能需要手动实现")
    print("   请使用以下方法之一:")
    print()
    print("   方法 1: 使用 Git patch")
    print("   git apply bug_16952_analysis/code/patch_server_args.py.patch")
    print("   git apply bug_16952_analysis/code/patch_ep_moe_layer.py.patch")
    print()
    print("   方法 2: 手动修改")
    print("   查看 bug_16952_analysis/code/fix_*.py 文件，手动应用修复")
    print()
    
    return 1

if __name__ == "__main__":
    exit(main())
