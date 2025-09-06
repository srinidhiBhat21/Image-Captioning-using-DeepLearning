#!/usr/bin/env python3
"""
Test script to verify all packages are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import torch
        print(f" PyTorch {torch.__version__} - OK")
        
        import torchvision
        print(f" TorchVision {torchvision.__version__} - OK")
        
        import transformers
        print(f" Transformers {transformers.__version__} - OK")
        
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("BLIP components - OK")
        
        import PIL
        print(f"Pillow {PIL.__version__} - OK")
        
        import matplotlib
        print(f"Matplotlib {matplotlib.__version__} - OK")
        
        import requests
        print(f"Requests {requests.__version__} - OK")
        
        import numpy
        print(f"NumPy {numpy.__version__} - OK")
        
        print("\n All packages installed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_torch():
    """Test PyTorch functionality"""
    import torch
    
    print(f"\n PyTorch Information:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   Running on CPU")
    
    # Test basic tensor operation
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x + y
    print(f"   Basic tensor operation: ")

if __name__ == "__main__":
    print("Testing BLIP Project Setup")
    print("=" * 40)
    
    if test_imports():
        test_torch()
        print("\nSetup verification complete!")
        print("You can now proceed to create the main project files.")
    else:
        print("\nSetup verification failed!")
        print("Please check package installations.")
