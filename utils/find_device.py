import torch


# TODO: implement to check the user of the process
def get_available_device():
    device = None
    if torch.cuda.is_available():
        min_memory = 1000.0 * 1024 * 1024  # 1000 MB
        selected_device = None
        # print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            # print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            # print(f"Free Memory: {free / 1e6:.2f} MB")
            # print(f"Used Memory: {used / 1e6:.2f} MB")
            # print(f"Total Memory: {total / 1e6:.2f} MB")
            if used < min_memory:
                selected_device = i
                break

        if selected_device is not None:
            device = torch.device(f"cuda:{selected_device}")

    elif torch.backends.mps.is_available():  # Apple M1/M2 support
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
