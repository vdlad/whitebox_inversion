import dataclasses

@dataclasses.dataclass
class Config:
    # Core
    prefix: str = "Hello! Will you marry me?"
    suffix_length: int = 13
    seed: int = 42*69*69
    batch_size: int = 500
    randomise: bool = False
    add_eos: bool = False
    relax_hot_val: float = 0.01
    model_id = "gpt2"

    # Learning
    learning_rate: float = 2e-3
    iterations: int = 500
    optimizer: str = "adam"
    scheduler_t_0: int = 28

    # Entropy projection
    start_entropy: float = 1.
    stop_entropy: float = 1.

    # Re-initialization
    reinit_threshold: int = 0
    reinit_rand_alpha: float = 1e-4
    reinit_blend_alpha: float = 1e-2

    # Blending
    best_blend_alpha: float = 0
    best_blend_threshold: float = 0.05

    # Discrete sampling
    discrete_sampling_temp: float = 2.0

    # Optuna
    use_optuna: bool = False
    optuna_trials: int = 10
    optuna_storage: str = "sqlite:///optuna.db"
    optuna_study_name: str = "whitebox-inversion"

    # Wandb
    wandb_logging: bool = True

def adapt_for_optuna(config: Config, trial):
    config.wandb_logging = False
    config.suffix_length = trial.suggest_int("suffix_length", 1, 30)
    config.relax_hot_val = trial.suggest_float("relax_hot_val", 0.001, 0.1)
    config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    config.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    config.scheduler_t_0 = trial.suggest_int("scheduler_t_0", 5, 30)
    config.stop_entropy = trial.suggest_float("stop_entropy", 0.99, 1.0)
    config.reinit_threshold = trial.suggest_int("reinit_threshold", 0, 300, step=10)
    config.best_blend_alpha = trial.suggest_float("best_blend_alpha", 0, 0.1)
    config.best_blend_threshold = trial.suggest_float("best_blend_threshold", 0, 0.1)
    config.discrete_sampling_temp = trial.suggest_float("discrete_sampling_temp", 1.0, 3.0)
    return config