import torch

if torch.cuda.is_available():
    # Create a tensor on the CPU
    cpu_tensor = torch.tensor([1, 2, 3])
    print(f"CPU tensor: {cpu_tensor}")

    # Move the tensor to the default ROCm device (gpu:0)
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"GPU tensor: {gpu_tensor}")

    # Perform operations on the GPU tensor
    result_tensor = gpu_tensor * 2
    print(f"Result on GPU: {result_tensor}")

    # Move a PyTorch model to the GPU
    # model = YourPyTorchModel()
    # model.to('cuda')
else:
    print("ROCm not available, operations will run on CPU.")