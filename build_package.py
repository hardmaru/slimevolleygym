#!/usr/bin/env python3
"""
Build script for FootsiesGym package.
This script builds the wheel and source distribution for the package.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔨 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Main build function."""
    print("🚀 Building FootsiesGym package...")
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    for pattern in ["build", "dist", "*.egg-info"]:
        if os.path.exists(pattern):
            import shutil
            shutil.rmtree(pattern, ignore_errors=True)
    
    # Install build dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "build", "twine"], 
                      "Installing build dependencies"):
        return False
    
    # Build the package
    if not run_command([sys.executable, "-m", "build"], 
                      "Building wheel and source distribution"):
        return False
    
    # Check the built packages
    if not run_command([sys.executable, "-m", "twine", "check", "dist/*"], 
                      "Checking built packages"):
        return False
    
    print("\n✅ Package built successfully!")
    print("📦 Built files:")
    
    dist_dir = Path("dist")
    if dist_dir.exists():
        for file in dist_dir.iterdir():
            print(f"  - {file.name}")
    
    print("\n📋 Next steps:")
    print("  1. Test installation: pip install dist/footsies_gym-*.whl")
    print("  2. Upload to PyPI: twine upload dist/*")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
