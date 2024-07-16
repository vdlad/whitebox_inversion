import torch
import torch.nn.functional as F
from .models import get_model_wrapper

def maximise_entropy_distribution_of_layer(activations, numerical_safety=1e-4):
    normaliser = torch.abs(activations).sum(dim=1).unsqueeze(1)
    p_activations = (torch.nn.functional.relu(activations) + 1e-4) / normaliser
    log_p_activations = torch.log(p_activations)
    losses = torch.nn.functional.kl_div(log_p_activations, 
                               (torch.ones(p_activations.shape[1])/p_activations.shape[1]).to(activations.device), 
                                        reduction='none')
    return losses.sum(dim=1)

def return_maximise_layer_entropy_distribution(numerical_safety=1e-4):
    _loss_wraper = lambda activations: maximise_entropy_distribution_of_layer(activations, numerical_safety=numerical_safety)
    return _loss_wraper

def maximise_entropy_distribution_of_logits(activations, unembedding_weights, numerical_safety=1e-4):
    logits = (unembedding_weights@activations.T).T
    normaliser = torch.abs(logits).sum()
    p_activations = (torch.nn.functional.relu(logits) + 1e-4) / normaliser
    log_p_activations = torch.log(p_activations)
    losses = torch.nn.functional.kl_div(log_p_activations, 
                               (torch.ones(p_activations.shape[1])/p_activations.shape[1]).to(activations.device), 
                                        reduction='none')
    return losses.sum(dim=1)

def return_maximise_logit_entropy_distribution(model_id, numerical_safety=1e-4, device="cuda"):
    _model, _ = get_model_wrapper(model_id, device=device)
    unembedding_weights = _model.lm_head.weight
    _loss_wraper = lambda activations: maximise_entropy_distribution_of_logits(activations, unembedding_weights, numerical_safety=numerical_safety)
    return _loss_wraper