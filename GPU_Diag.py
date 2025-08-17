#!/usr/bin/env python3
"""
CUDA Diagnostic Script
This script helps diagnose CUDA initialization issues
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("=== NVIDIA Driver Check ===")
    returncode, stdout, stderr = run_command("nvidia-smi")
    if returncode == 0:
        print("✓ NVIDIA driver installed")
        print(f"Driver output:\n{stdout}\n")
        return True
    else:
        print("✗ NVIDIA driver not found or not working")
        print(f"Error: {stderr}")
        return False

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    print("=== CUDA Toolkit Check ===")
    
    # Check nvcc
    returncode, stdout, stderr = run_command("nvcc --version")
    if returncode == 0:
        print("✓ CUDA toolkit (nvcc) found")
        print(f"NVCC version:\n{stdout}\n")
    else:
        print("✗ CUDA toolkit (nvcc) not found in PATH")
    
    # Check common CUDA paths
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.2"
    ]
    
    print("Checking common CUDA installation paths:")
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✓ Found CUDA at: {path}")
        else:
            print(f"✗ Not found: {path}")
    
    # Check environment variables
    print(f"\nEnvironment variables:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print()

def check_python_cuda_libs():
    """Check Python CUDA libraries"""
    print("=== Python CUDA Libraries Check ===")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
    except ImportError:
        print("✗ PyTorch not installed")
    except Exception as e:
        print(f"✗ PyTorch CUDA error: {e}")
    
    # Check CuPy
    try:
        import cupy as cp
        print(f"✓ CuPy installed: {cp.__version__}")
        
        # Try basic operations
        try:
            device = cp.cuda.Device()
            print(f"  Current device: {device.id}")
            print(f"  Device name: {device.compute_capability}")
            
            # Try creating an array (this triggers CURAND)
            test_array = cp.random.randn(10, 10)
            print("✓ CuPy random number generation working")
            
        except Exception as e:
            print(f"✗ CuPy device/random error: {e}")
            
    except ImportError:
        print("✗ CuPy not installed")
    except Exception as e:
        print(f"✗ CuPy import error: {e}")
    
    print()

def check_gpu_memory():
    """Check GPU memory usage"""
    print("=== GPU Memory Check ===")
    returncode, stdout, stderr = run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    if returncode == 0:
        lines = stdout.strip().split('\n')
        for i, line in enumerate(lines):
            used, total = line.split(', ')
            print(f"GPU {i}: {used}MB / {total}MB used")
    else:
        print("Could not query GPU memory")
    print()

def check_cuda_processes():
    """Check what processes are using CUDA"""
    print("=== CUDA Processes Check ===")
    returncode, stdout, stderr = run_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    if returncode == 0:
        print("CUDA processes:")
        print(stdout)
    else:
        print("Could not query CUDA processes")
    print()

def main():
    print("CUDA Diagnostic Tool")
    print("===================\n")
    
    # Run all checks
    driver_ok = check_nvidia_driver()
    check_cuda_toolkit()
    check_python_cuda_libs()
    
    if driver_ok:
        check_gpu_memory()
        check_cuda_processes()
    
    print("=== Recommendations ===")
    print("Common fixes for CURAND_STATUS_INITIALIZATION_FAILED:")
    print("1. Restart your system (driver/CUDA issues)")
    print("2. Check if another process is using all GPU memory")
    print("3. Update NVIDIA drivers")
    print("4. Reinstall CUDA toolkit")
    print("5. Reinstall CuPy: pip uninstall cupy && pip install cupy-cuda12x")
    print("6. Set CUDA_VISIBLE_DEVICES=0 if you have multiple GPUs")
    print("7. Run as root/administrator if permission issues")

if __name__ == "__main__":
    main()