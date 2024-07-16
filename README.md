```
whitebox-inversion/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── attack.py
│   ├── utils.py
│   └── loss_functions.py
├── scripts/
│   └── run_attack.py
├── requirements.txt
├── README.md
└── .gitignore
```


# Whitebox Inversion

This project implements a white-box inversion attack on language models.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/whitebox-inversion.git
   cd whitebox-inversion
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the attack:

```
python scripts/run_attack.py
```

You can modify the configuration in `src/config.py` to adjust the attack parameters.

## Project Structure

- `src/`: Contains the main source code
  - `config.py`: Configuration class and Optuna adaptation
  - `models.py`: Model handling functions
  - `attack.py`: Main attack function
  - `utils.py`: Utility functions
  - `loss_functions.py`: Loss function implementations
- `scripts/`: Contains runnable scripts
  - `run_attack.py`: Main script to run the attack
- `requirements.txt`: List of required packages
- `README.md`: This file

## License

[Your chosen license]