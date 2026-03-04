#!/usr/bin/env python3
"""
Setup Script - Creates .env from .env.example if it doesn't exist.

Run this before starting the API for the first time:
    python setup_env.py

Author: Deep Retina Grade Project
Date: February 2026
"""

import shutil
from pathlib import Path


def setup_env():
    """Create .env from .env.example if it doesn't exist."""
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if env_file.exists():
        print("✅ .env already exists")
        return
    
    if not env_example.exists():
        print("❌ .env.example not found!")
        return
    
    shutil.copy(env_example, env_file)
    print(f"✅ Created .env from .env.example")
    print(f"   Edit {env_file} to customize settings")
    print(f"   Key settings:")
    print(f"     DR_DEMO_MODE=false       (set true for quick demos)")
    print(f"     DR_QUALITY_THRESHOLD=0.4 (image quality gate)")
    print(f"     DR_MC_SAMPLES=5          (uncertainty samples)")
    print(f"     DR_ENSEMBLE_ENABLED=false (multi-model ensemble)")


if __name__ == "__main__":
    setup_env()
