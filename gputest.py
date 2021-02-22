import torch

x = torch.cuda.current_device()

print(torch.cuda.device(x))

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(x))

print(torch.cuda.is_available())