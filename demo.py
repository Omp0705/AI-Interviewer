import torch
print("CUDA Available:", torch.cuda.is_available())
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())


print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print number of GPUs
print(torch.cuda.get_device_name(0)) 