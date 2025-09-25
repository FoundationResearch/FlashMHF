import torch
import torch.nn as nn
from flash_ffn_moe_torch import FlashFFNMoE

# Shapes analogous to Triton version
batch_size = 2
num_heads = 4
seq_len = 192*2
head_dim = 128
num_kuv_heads = 4
num_expert = 2
expert_dim = 128

device = 'cuda'

def t(shape, requires_grad=False):
    return torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)

def torch_ground_truth(Q, K, U, V, R):
    O = torch.zeros_like(Q)
    for i in range(num_expert):
        oi = (nn.functional.silu(Q @ K[:, i, :, :].transpose(1, 2)) * (Q @ U[:, i, :, :].transpose(1, 2))) @ (V[:, i, :, :])
        O += oi * R[:, :, :, i:i+1]
    return O


Q = t((batch_size, num_heads, seq_len, head_dim), requires_grad=True)
K = t((num_kuv_heads, num_expert, expert_dim, head_dim), requires_grad=True)
U = t((num_kuv_heads, num_expert, expert_dim, head_dim), requires_grad=True)
V = t((num_kuv_heads, num_expert, expert_dim, head_dim), requires_grad=True)
R = torch.randn((batch_size, num_heads, seq_len, num_expert), device=device, dtype=torch.float32, requires_grad=True)

model = FlashFFNMoE().to(device)
print("[Python] Launching Kernel")
O = model(Q, K, U, V, R)
print("O.mean(): ", O.mean().item())
G = torch_ground_truth(Q, K, U, V, R)
print('forward max diff:', (O - G).abs().max().item(), 'original max', G.abs().max().item())

loss1 = (O * 2.0).sum()
loss1.backward()
dQ, dK, dU, dV, dR = Q.grad.clone(), K.grad.clone(), U.grad.clone(), V.grad.clone(), R.grad.clone()

# clear grad
Q.grad.zero_()
K.grad.zero_()
U.grad.zero_()
V.grad.zero_()
R.grad.zero_()

loss2 = (G * 2.0).sum()
loss2.backward()

print('dQ max diff:', (dQ - Q.grad).abs().max().item(), 'original max', Q.grad.abs().max().item(), 'percentage', 100*(dQ - Q.grad).abs().max().item() / Q.grad.abs().max().item(), "%")
print('dK max diff:', (dK - K.grad).abs().max().item(), 'original max', K.grad.abs().max().item(), 'percentage', 100*(dK - K.grad).abs().max().item() / K.grad.abs().max().item(), "%")
print('dU max diff:', (dU - U.grad).abs().max().item(), 'original max', U.grad.abs().max().item(), 'percentage', 100*(dU - U.grad).abs().max().item() / U.grad.abs().max().item(), "%")
print('dV max diff:', (dV - V.grad).abs().max().item(), 'original max', V.grad.abs().max().item(), 'percentage', 100*(dV - V.grad).abs().max().item() / V.grad.abs().max().item(), "%")
print('dR max diff:', (dR - R.grad).abs().max().item(), 'original max', R.grad.abs().max().item(), 'percentage', 100*(dR - R.grad).abs().max().item() / R.grad.abs().max().item(), "%")