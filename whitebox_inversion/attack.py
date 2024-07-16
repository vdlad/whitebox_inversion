import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from typing import Callable
from .utils import to_relaxed_one_hot, simplex_projection, entropy_projection

def attack(model, tokenizer, config, loss_func: Callable, verbose: bool = False, discrete_loss_sample_rate: int = 1):
    device = next(model.parameters()).device
    
    if config.optimizer == "adamw":
        optimizer = AdamW([torch.zeros(1, device=device)], lr=config.learning_rate)
    elif config.optimizer == "adam":
        optimizer = Adam([torch.zeros(1, device=device)], lr=config.learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {config.optimizer}")

    prefix_str = config.prefix if config.prefix else ""

    suffix_tokens = torch.randint(low=0, high=tokenizer.vocab_size, size=(config.batch_size, config.suffix_length), device="cpu")
    prefix_tokens = tokenizer.encode(prefix_str, return_tensors='pt').squeeze(0).repeat(config.batch_size, 1).to("cpu")

    all_tokens = torch.cat([prefix_tokens, suffix_tokens], dim=1).to(device)
    suffix_slice = slice(prefix_tokens.shape[1], all_tokens.shape[1])

    labels = all_tokens.clone().type(torch.int64)
    labels[:, :suffix_slice.stop] = -100
    labels = labels[:,1:].flatten(start_dim=0,end_dim=1)

    inputs = to_relaxed_one_hot(all_tokens, tokenizer.vocab_size, hot_val=config.relax_hot_val)
    if config.randomise:
        random_values = torch.rand_like(inputs[:,suffix_slice])
        normalized_values = random_values / random_values.sum(dim=-1, keepdim=True)
        inputs[:,suffix_slice] = normalized_values

    inputs.requires_grad_()

    optimizer.param_groups.clear()
    optimizer.add_param_group({"params": [inputs]})

    batch_size, prediction_seq_length, vocab_size = config.batch_size, \
    inputs.shape[1] - 1, tokenizer.vocab_size

    scheduler = CosineAnnealingLR(optimizer, config.scheduler_t_0)

    best_loss = torch.inf * torch.ones(1, device=device)
    best_discrete = None
    current_entropy = config.start_entropy
    entropy_delta = (config.stop_entropy - config.start_entropy) / config.iterations

    for i in tqdm(range(1, config.iterations + 1)):
        input_embeds = (inputs @ model.transformer.wte.weight).to(device)
        activations = (model(inputs_embeds=input_embeds))[:, -1, :]
        loss = loss_func(activations)

        optimizer.zero_grad()
        dummy_grad = torch.ones_like(loss)
        loss.backward(gradient=dummy_grad)

        inputs.grad.data[:, : suffix_slice.start] = 0
        inputs.grad.data[:, suffix_slice.stop :] = 0

        optimizer.step()
        scheduler.step()

        inputs.data[:, suffix_slice] = simplex_projection(inputs.data[:, suffix_slice])
        if current_entropy != 1.0:
            inputs.data[:, suffix_slice] = entropy_projection(inputs.data[:, suffix_slice], current_entropy)
        current_entropy += entropy_delta
        discrete = torch.argmax(inputs.data[:, suffix_slice], dim=2)
        all_tokens[:, suffix_slice] = discrete
        if verbose or config.wandb_logging:
          save_loss = loss.detach().cpu().mean()
        del activations, loss

        if i % discrete_loss_sample_rate == 0:
            with torch.no_grad():
                activations_discrete = (model(all_tokens))[:, -1]
                discrete_loss = loss_func(activations_discrete)
                avg_loss_i, avg_best_loss = discrete_loss.mean(), best_loss.mean()
                if avg_loss_i < avg_best_loss:
                    best_loss = discrete_loss
                    best_discrete = discrete
            if verbose:
                current_discrete_text = [tokenizer.decode(x) for x in discrete[:3]]
                print(f"[{i}] L-rel: {save_loss:.5f} / L-dis: {discrete_loss.flatten().mean():.5f} / Best: {best_loss.flatten().mean():.5f}")
                print(f" |- Curr: {current_discrete_text}")

            if config.wandb_logging:
                wandb.log({
                    "iteration": i,
                    "relaxed_loss": save_loss,
                    "avg_discrete_loss": avg_best_loss.item(),
                    "avg_best_loss": avg_best_loss.item(),
                    "current_entropy": current_entropy,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
            del activations_discrete, discrete_loss

    return best_loss, best_discrete