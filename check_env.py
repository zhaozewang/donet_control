#!/usr/bin/env python3
"""
Environment check script - verifies conda environment and dependencies.
Run this before running the main training or tests.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_conda_env():
    """Check if we're in the right conda environment."""
    conda_env = None
    
    # Check CONDA_DEFAULT_ENV
    import os
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if conda_env:
        print(f"Conda environment: {conda_env}")
        if conda_env == 'mujoco_env':
            print("✅ In correct conda environment")
            return True
        else:
            print("⚠️  Not in 'mujoco_env' environment")
            print("Run: conda activate mujoco_env")
            return False
    else:
        print("⚠️  No conda environment detected")
        print("Run: conda activate mujoco_env")
        return False


def check_dependencies():
    """Check if key dependencies are installed."""
    required_packages = [
        'torch',
        'numpy', 
        'scipy',
        'hydra',
        'omegaconf',
        'wandb',
        'tqdm',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All core dependencies available")
        return True


def check_project_structure():
    """Check that project files exist."""
    required_files = [
        'main.py',
        'requirements.txt',
        'config/config.yaml',
        'src/models/gru_remi.py',
        'src/tasks/humanoid_walking.py',
        'src/training/curriculum_trainer.py',
        'assets/humanoid.xml'
    ]
    
    missing_files = []
    base_path = Path(__file__).parent
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ Project structure complete")
        return True


def main():
    """Run all environment checks."""
    print("=" * 60)
    print("REMI Humanoid Control - Environment Check")
    print("=" * 60)
    
    all_good = True
    
    print("\n🐍 Python Version Check")
    print("-" * 30)
    all_good &= check_python_version()
    
    print("\n🐍 Conda Environment Check")
    print("-" * 30)
    all_good &= check_conda_env()
    
    print("\n📦 Dependencies Check")
    print("-" * 30)
    all_good &= check_dependencies()
    
    print("\n📁 Project Structure Check")
    print("-" * 30)
    all_good &= check_project_structure()
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("🎉 Environment setup complete!")
        print("\nNext steps:")
        print("1. Run test: python3 test_setup.py")
        print("2. Run training: python3 main.py")
    else:
        print("❌ Environment setup incomplete")
        print("\nPlease fix the issues above before proceeding.")
        
        if not check_conda_env():
            print("\n💡 Quick fix:")
            print("conda create -n mujoco_env python=3.9")
            print("conda activate mujoco_env")
            print("pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)