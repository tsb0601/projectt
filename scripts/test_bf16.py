import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()

minimum_bf16_value = 0.0078125

preturbed_bf16_value = minimum_bf16_value - 0.0001

print(f"minimum_bf16_value: {minimum_bf16_value}, preturbed_bf16_value: {preturbed_bf16_value}")

minimum_bf16_tensor = torch.tensor([minimum_bf16_value], dtype=torch.bfloat16, device=device)
preturbed_bf16_tensor = torch.tensor([preturbed_bf16_value], dtype=torch.bfloat16, device=device)
print(f"minimum_bf16_tensor: {minimum_bf16_tensor.item()}, preturbed_bf16_tensor: {preturbed_bf16_tensor.item()}")

above1_bf16_value = minimum_bf16_value + 1
above1_preturbed_value = preturbed_bf16_value + 1
above1_bf16_tensor = torch.tensor([above1_bf16_value], dtype=torch.bfloat16, device=device)
above1_preturbed_tensor = torch.tensor([above1_preturbed_value], dtype=torch.bfloat16, device=device)
print(f"above1_bf16_tensor: {above1_bf16_tensor.item()}, above1_precise_tensor: {above1_preturbed_tensor.item()}")

cpu_minimum_bf16_tensor = torch.tensor([minimum_bf16_value], dtype=torch.bfloat16)
cpu_preturbed_bf16_tensor = torch.tensor([preturbed_bf16_value], dtype=torch.bfloat16)
print(f"cpu_minimum_bf16_tensor: {cpu_minimum_bf16_tensor.item()}, cpu_preturbed_bf16_tensor: {cpu_preturbed_bf16_tensor.item()}")

cpu_above1_bf16_tensor = torch.tensor([above1_bf16_value], dtype=torch.bfloat16)
cpu_above1_preturbed_tensor = torch.tensor([above1_preturbed_value], dtype=torch.bfloat16)
print(f"cpu_above1_bf16_tensor: {cpu_above1_bf16_tensor.item()}, cpu_above1_preturbed_tensor: {cpu_above1_preturbed_tensor.item()}")