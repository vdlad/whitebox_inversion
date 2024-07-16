import torch

def to_relaxed_one_hot(tokens: torch.Tensor, vocab_size: int, hot_val: float = 1.0) -> torch.Tensor:
    batch_size, seq_len = tokens.size()
    one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=tokens.device)
    one_hot.scatter_(2, tokens.unsqueeze(-1).to(torch.int64), hot_val)
    remaining_prob = hot_val / (vocab_size - 1)
    one_hot += remaining_prob * (1 - one_hot)
    return one_hot.to(tokens.device)

def simplex_projection(tensor: torch.Tensor) -> torch.Tensor:
    d_batch, d_tokens, d_vocab = tensor.shape
    mu, _ = torch.sort(tensor, descending=True, dim=-1)
    cumulative = mu.cumsum(dim=-1)
    indices = torch.arange(1, d_vocab + 1, device=tensor.device).expand(d_batch, d_tokens, -1).float()
    threshold = (cumulative - 1) / indices
    rho = (mu > threshold).sum(dim=-1) - 1
    rho = torch.clamp(rho, 0, d_vocab - 1)
    threshold_per_row = torch.gather(threshold, 2, rho.unsqueeze(-1))
    projected = torch.max(tensor - threshold_per_row, torch.zeros_like(tensor))
    return projected

def entropy_projection(tensor: torch.Tensor, entropy: float) -> torch.Tensor:
    positive_mask = (tensor > 0).float()
    positive_count = positive_mask.sum(dim=-1, keepdim=True)
    c = positive_mask / positive_count
    R = torch.sqrt(1 - entropy - 1 / (positive_count))
    if R.isnan().any():
        return tensor
    norm_s_c = torch.norm(tensor - c, dim=-1, keepdim=True)
    needs_projection = (norm_s_c < R).float()
    does_not_need_projection = 1 - needs_projection
    scaled_s = torch.where(needs_projection.bool(), (R / norm_s_c) * (tensor - c) + c, tensor)
    projection = simplex_projection(scaled_s)
    result = does_not_need_projection * tensor + needs_projection * projection
    return result