import torch
import wandb
import optuna
import dataclasses
from src.config import Config, adapt_for_optuna
from src.models import get_model_wrapper
from src.attack import attack
from src.loss_functions import return_maximise_logit_entropy_distribution

PROJECT = "whitebox-inversion"

def main():
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.wandb_logging and not config.use_optuna:
        wandb.init(project=PROJECT, config=dataclasses.asdict(config))

    print("[+] Loading model and tokenizer...")
    t_model, t_toke = get_model_wrapper(config.model_id, 6)
    loss_e = return_maximise_logit_entropy_distribution(config.model_id)

    if config.use_optuna:
        print("[+] Using Optuna ...")
        study = optuna.create_study(
            study_name=config.optuna_study_name,
            storage=config.optuna_storage,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
        )
        study.optimize(
            lambda trial: attack(t_model, t_toke, adapt_for_optuna(config, trial), loss_e)[0].mean().item(),
            n_trials=config.optuna_trials,
        )
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        print("[+] Start Attack ...")
        loss, best_prompt = attack(t_model, t_toke, config, loss_e, verbose=config.wandb_logging, discrete_loss_sample_rate=1)
        print()
        print("[+] Done. Final loss:", loss.mean())
        print("[*] Done. Example best prompt:", t_toke.decode(best_prompt[0]))
        print()

    if config.wandb_logging:
        wandb.finish()

    return best_prompt

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()