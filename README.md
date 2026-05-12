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
To use the library, open this folder in your terminal. The base command is:

```bash
am-tools --help
```

> Note: `amtools` without the dash also works.

### `am-tools lint`
There are two subcommands in `am-tools lint`:

```bash
am-tools lint schema
am-tools lint sidecar
```
#### Sidecar Schemas
When you run `am-tools lint schema` it will prompt you to give a path to your schema file. For this library, a schema helps the program know the names of missing columns, or to rename columns into something more readable ("Latitude" instead of "Col3" for example). There are two acceptable formats for schemas: one for sidecars that are missing a header entirely (`column_names`) and one for when there already are column names but they need to be renamed (`column_name_mapping`).

```json
{
  "column_names": [
    "Filename",
    "CreateDate",
    "GPSAltitude",
  ]
}
```
or 
```json
{
  "column_name_mapping": {
    "Col1":"NewColumnName1",
    "ExampleCol2":"GPSLongitude"
  }
}
```

**If you include the `column_names` section, it will assume that there is no header row. If there is a header row and you give it a schema with this section, it will read your header as the first row of data and break everything.** 

For `column_name_mapping`, it will look for columns with names on the left side of the colon and rename them to the right side of the colon. You don't need to put in a mapping for every column, just the ones you want to rename. 

After running `am-tools lint schema`, if any errors are detected, it will tell you how to fix them. This is just a quick check and might not catch errors until we run it with the sidecar. Thankfully, you can lint a sidecar as well.

#### Linting a sidecar


### `am-tools sidecar`


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
