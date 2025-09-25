import torch
import flash_ffn_moe as _ext


class FlashFFNMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, U, V, R):
        O = torch.empty_like(Q)
        R = R.transpose(2,3).contiguous()
        _ext.flashffn_moe_forward(Q, K, U, V, R, O)
        ctx.save_for_backward(Q, K, U, V, R)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, U, V, R = ctx.saved_tensors
        dQ = torch.zeros(Q.shape, dtype=torch.float, device=Q.device)
        dK = torch.zeros(K.shape, dtype=torch.float, device=K.device)
        dU = torch.zeros(U.shape, dtype=torch.float, device=U.device)
        dV = torch.zeros(V.shape, dtype=torch.float, device=V.device)
        dR = torch.zeros(R.shape, dtype=torch.float, device=R.device)
        _ext.flashffn_moe_backward_intermediate_atomicqr(Q,K,U,V,R,dQ,dK,dU,dV,dR,grad_output.contiguous())
        dR = dR.transpose(2,3).contiguous()
        return dQ, dK, dU, dV, dR


class FlashFFNMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, U, V, R):
        return FlashFFNMoEFunction.apply(Q, K, U, V, R) 