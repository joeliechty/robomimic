import torch
import torch.nn.functional as F
import psutil
import os
import numpy as np

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_clip_loading():
    """Test CLIP model loading and memory usage"""
    
    print("=" * 60)
    print("TESTING CLIP MODEL LOADING")
    print("=" * 60)
    
    # Baseline memory
    mem_start = get_memory_usage()
    print(f"\n1. Baseline memory: {mem_start:.1f} MB")
    
    # Load CLIP on CPU
    print("\n2. Loading CLIP on CPU...")
    import clip
    try:
        clip_model_cpu, _ = clip.load("ViT-B/32", device="cpu")
        clip_model_cpu.eval()
        mem_after_cpu = get_memory_usage()
        print(f"   ✓ CLIP loaded on CPU")
        print(f"   Memory used: {mem_after_cpu - mem_start:.1f} MB")
    except Exception as e:
        print(f"   ✗ Failed to load CLIP on CPU: {e}")
        return
    
    # Try loading on GPU if available
    if torch.cuda.is_available():
        print("\n3. Loading CLIP on GPU...")
        try:
            clip_model_gpu, _ = clip.load("ViT-B/32", device="cuda")
            clip_model_gpu.eval()
            mem_after_gpu = get_memory_usage()
            print(f"   ✓ CLIP loaded on GPU")
            print(f"   CPU memory used: {mem_after_gpu - mem_start:.1f} MB")
            print(f"   GPU memory used: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"   ✗ Failed to load CLIP on GPU: {e}")
    
    # Test encoding
    print("\n4. Testing image encoding...")
    
    # Create dummy image (1, 3, 224, 224)
    dummy_img = torch.randn(1, 3, 224, 224)
    
    # Normalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    dummy_img = (dummy_img - mean) / std
    
    # Test on CPU
    print("   Testing on CPU...")
    mem_before_encode = get_memory_usage()
    with torch.no_grad():
        feats_cpu = clip_model_cpu.encode_image(dummy_img)
    mem_after_encode = get_memory_usage()
    print(f"   ✓ Encoding successful")
    print(f"   Features shape: {feats_cpu.shape}")
    print(f"   Memory used for encoding: {mem_after_encode - mem_before_encode:.1f} MB")
    
    # Clean up
    del dummy_img, feats_cpu
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Test repeated encoding (simulating rollout)
    print("\n5. Testing repeated encoding (simulating rollout)...")
    mem_peak = get_memory_usage()
    print(f"   Peak memory before rollout sim: {mem_peak:.1f} MB")
    
    for i in range(10):
        dummy_img = torch.randn(1, 3, 224, 224)
        dummy_img = (dummy_img - mean) / std
        with torch.no_grad():
            feats = clip_model_cpu.encode_image(dummy_img)
        del dummy_img, feats
        
        if (i + 1) % 5 == 0:
            mem = get_memory_usage()
            print(f"   After {i+1} iterations: {mem:.1f} MB (delta: {mem - mem_peak:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("✓ CLIP MEMORY TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_clip_loading()
