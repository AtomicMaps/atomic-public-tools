# atomic-public-tools

Welcome to the public-facing metadata sidecar repository for Atomic Maps. This repository contains some helpful libraries and functions to help you format your sidecar metadata so that it fits our specifications. Additionally, this repository contains some sample data (both clean and not clean) so you can familiarize yourself with the tool before pointing it at your data.

The main tool of this repo is the the library `am-tools`. It is designed to validate and format metadata files (JSON, CSV, Excel) before submitting them for Flow intake. More detailed instructions on how to use the library are below.

## Install

Requires Python 3.10+. In terminal or command prompt, open this folder and run the following command.

```bash
pip install -e ".[dev]"
```
This will install both `am-tools` as well as all requirements for `am-tools`. 

## Usage
To use the library, open this folder in command prompt

```bash
am-tools --help
am-tools metadata --help
am-tools metadata validate path/to/metadata.json
am-tools metadata format path/to/metadata.csv --output formatted.json
```

You can also run the package directly:

```bash
python -m atomic_tools --help
```

## Project layout

```
src/atomic_tools/
├── cli.py              # root Typer app
├── commands/           # one module per subcommand group
├── schemas/            # Pydantic models for the canonical metadata shape
├── formatters/         # per-format normalizers (json/csv/excel)
└── io/                 # file readers
tests/                  # pytest suite
```

## Development

```bash
pytest
ruff check src tests
```
