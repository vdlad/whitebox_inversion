import torch

def to_relaxed_one_hot(tokens: torch.Tensor, vocab_size: int, hot_val: float = 1.0) -> torch.Tensor:
    device = tokens.device
    batch_size, seq_len = tokens.size()    
    # Calculate the 'cold' value
    cold_val = hot_val / (vocab_size - 1)    
    # Create the tensor with the 'cold' value
    one_hot = torch.full((batch_size, seq_len, vocab_size),
                         fill_value=cold_val,
                         device=device, dtype=torch.float32)    
    # Use scatter to set the 'hot' values, factoring in normalisation condition
    one_hot.scatter_(2, tokens.unsqueeze(-1), cold_val + (1.0-cold_val)*hot_val)    
    return one_hot

def simplex_projection(tensor: torch.Tensor) -> torch.Tensor:
    d_batch, d_sequence, d_dims = tensor.shape
    mu, _ = torch.sort(tensor, descending=True, dim=-1)
    cumulative = mu.cumsum(dim=-1)
    indices = torch.arange(1, d_dims + 1, device=tensor.device).expand(d_batch, d_sequence, -1)
    threshold = (cumulative - 1) / indices
    rho = torch.clamp((mu > threshold).sum(dim=-1) - 1, 0, d_dims-1)
    threshold_per_row = torch.gather(threshold, 2, rho.unsqueeze(-1))
    return torch.maximum(tensor - threshold_per_row, torch.zeros(1, dtype=tensor.dtype, device=tensor.device))

def entropy_projection(tensor: torch.Tensor, entropy: float) -> torch.Tensor:
    original_shape = tensor.shape
    s = tensor.view(original_shape[0], -1, original_shape[-1])
    positive_mask = (s > 0).float()
    positive_count = positive_mask.sum(dim=-1, keepdim=True)
    c = positive_mask / positive_count
    R = torch.sqrt(1 - entropy - 1 / (positive_count))
    if R.isnan().any():
        return tensor
    norm_s_c = torch.norm(s - c, dim=-1, keepdim=True)
    needs_projection = (norm_s_c < R).float()
    does_not_need_projection = 1 - needs_projection
    scaled_s = torch.where(needs_projection.bool(), (R / norm_s_c) * (s - c) + c, s)
    projection = simplex_projection(scaled_s)
    result = does_not_need_projection * s + needs_projection * projection
    return result.view(original_shape)

