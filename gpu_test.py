import torch

# Check if CUDA is available
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    
    # Get information about each GPU
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {gpu_props.name}")
        print(f"Total memory: {gpu_props.total_memory / 1024**2:.1f} MB")
        print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")
        print(f"Multi-processor count: {gpu_props.multi_processor_count}")
        
    # Get current GPU device
    current_device = torch.cuda.current_device()
    print(f"\nCurrent GPU device: {current_device}")
    
    # Get current GPU memory usage
    print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Current memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
else:
    print("No GPU available. Using CPU only.")
