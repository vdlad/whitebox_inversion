from setuptools import setup, find_packages

setup(
    name="whitebox_inversion",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "optuna",
        "wandb",
    ],
)