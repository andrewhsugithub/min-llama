import torch


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


def tensor_memory_size(tensor):
    """
    Calculate the memory size of a given tensor in bytes and gigabytes.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        dict: A dictionary containing memory size in bytes and gigabytes.
    """
    element_size = tensor.element_size()  # Size of a single element in bytes
    total_elements = tensor.numel()  # Total number of elements in the tensor
    total_memory_bytes = total_elements * element_size  # Total memory in bytes
    total_memory_gb = total_memory_bytes / (1024**3)  # Convert to gigabytes

    return total_memory_gb
