# atomic-public-tools

Client-facing tools to interact with Flow. `am-tools` helps you validate and format metadata files (JSON, CSV, Excel) before submitting them for Flow intake.

## Install

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
```

## Usage

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
