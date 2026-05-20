import torch

def spectral_projection(K_tensor, s_max=0.97):
  with torch.no_grad():
    K_cpu = K_tensor.detach().cpu()
    try:
      U_K, S_K, V_K = torch.linalg.svd(K_cpu, full_matrices=False)
    except Exception:
      U_K, S_K, V_K = torch.linalg.svd(K_cpu)
    S_clamp = torch.clamp(S_K, max=s_max)
    K_proj = (U_K @ torch.diag_embed(S_clamp) @ V_K)
    K_tensor.data.copy_(K_proj.to(K_tensor.device))