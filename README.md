# Whitebox Inversion

This project implements a white-box inversion attack on language models.

## Project Structure

```
whitebox_inversion/
├── whitebox_inversion/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── attack.py
│   ├── utils.py
│   └── loss_functions.py
├── scripts/
│   └── run_attack.py
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/vdlad/whitebox_inversion.git
   cd whitebox_inversion
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the project in editable mode:
   ```
   pip install -e .
   ```

   This will install all required dependencies and the project itself.

## Usage

To run the attack:

```
python scripts/run_attack.py
```

You can modify the configuration in `whitebox_inversion/config.py` to adjust the attack parameters.

## Development

If you're developing this project:

1. Install development dependencies:
   ```
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```
   pytest
   ```

3. Check code style:
   ```
   flake8
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT]