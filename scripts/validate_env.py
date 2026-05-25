"""
EXP-000: Environment Validation Script
Verifies that all dependencies are installed and GPU is operational.
"""

import sys
import platform
import json
from pathlib import Path
from datetime import datetime


def check_python():
    """Check Python version."""
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")
    assert v.major == 3 and v.minor >= 10, f"Python 3.10+ required, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def check_torch():
    """Check PyTorch and CUDA."""
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / 1e9
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {gpu_mem:.1f} GB")
        print(f"BF16 support: {torch.cuda.is_bf16_supported()}")
    else:
        gpu_mem = None
        print("WARNING: CUDA not available. Training will be very slow on CPU.")

    return {
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(gpu_mem, 1) if gpu_mem else None,
        "bf16_support": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }


def check_forward_backward():
    """Run a forward + backward pass on a synthetic IMU-shaped tensor."""
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate a batch of IMU windows: (batch=8, time=500, channels=6)
    x = torch.randn(8, 500, 6, device=device)

    # Simple encoder: linear projection of patches
    patch_len = 25
    num_patches = 500 // patch_len  # 20
    x_patches = x.reshape(8, num_patches, patch_len * 6)  # (8, 20, 150)

    model = nn.Sequential(
        nn.Linear(patch_len * 6, 384),
        nn.GELU(),
        nn.Linear(384, 384),
    ).to(device)

    out = model(x_patches)  # (8, 20, 384)
    loss = out.mean()
    loss.backward()

    print(f"Forward/backward pass OK on {device}")
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Patches:      {list(x_patches.shape)}")
    print(f"  Output shape: {list(out.shape)}")
    print(f"  Loss:         {loss.item():.6f}")
    return True


def check_mixed_precision():
    """Verify torch.cuda.amp works."""
    import torch

    if not torch.cuda.is_available():
        print("Skipping mixed-precision check (no GPU)")
        return False

    device = "cuda"
    x = torch.randn(8, 20, 150, device=device)
    model = torch.nn.Linear(150, 384).to(device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)
        loss = out.mean()

    loss.backward()
    print(f"Mixed-precision (FP16) OK. Output dtype: {out.dtype}")
    return True


def check_dependencies():
    """Check that all key packages are importable."""
    deps = {}
    packages = [
        "numpy", "scipy", "pandas", "sklearn", "h5py",
        "yaml", "omegaconf", "matplotlib", "seaborn",
        "tqdm", "einops", "pytest",
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "OK")
            deps[pkg] = ver
            print(f"  {pkg}: {ver}")
        except ImportError:
            deps[pkg] = "MISSING"
            print(f"  {pkg}: MISSING !!!")
    return deps


def main():
    print("=" * 60)
    print("IMU-JEPA Environment Validation (EXP-000)")
    print("=" * 60)
    print()

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    # 1. Python
    print("--- Python ---")
    env_info["python"] = check_python()
    print()

    # 2. Dependencies
    print("--- Dependencies ---")
    env_info["dependencies"] = check_dependencies()
    print()

    # 3. PyTorch + CUDA
    print("--- PyTorch & CUDA ---")
    env_info["torch"] = check_torch()
    print()

    # 4. Forward/backward
    print("--- Forward/Backward Pass ---")
    env_info["forward_backward_ok"] = check_forward_backward()
    print()

    # 5. Mixed precision
    print("--- Mixed Precision ---")
    env_info["mixed_precision_ok"] = check_mixed_precision()
    print()

    # Save environment snapshot
    out_dir = Path("experiments/exp_000_env_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    env_path = out_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2, default=str)
    print(f"Environment snapshot saved to: {env_path}")

    # Summary
    print()
    print("=" * 60)
    missing = [k for k, v in env_info.get("dependencies", {}).items() if v == "MISSING"]
    if missing:
        print(f"FAIL: Missing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
    elif not env_info.get("torch", {}).get("cuda_available", False):
        print("WARN: No CUDA GPU detected. Training will be CPU-only.")
    else:
        print("ALL CHECKS PASSED. Environment is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
