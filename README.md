# atomic-public-tools

Welcome to the public-facing metadata sidecar repository for Atomic Maps. This repository contains some helpful libraries and functions to help you format your sidecar metadata so that it fits our specifications. Additionally, this repository contains some sample data (both clean and not clean) so you can familiarize yourself with the tool before pointing it at your data.

The main tool of this repo is the the library `am-tools`. It is designed to validate and format metadata files (JSON, CSV, Excel) before submitting them for Flow intake. More detailed instructions on how to use the library are below.

## Install

### Prerequisites

`am-tools` shells out to two external tools for metadata extraction. Install them before running `pip install`:

- **exiftool** — required for image and video metadata.
    - macOS: `brew install exiftool`
    - Ubuntu/Debian: `sudo apt install libimage-exiftool-perl`
    - Windows: download from https://exiftool.org/
- **pdal** — required only if you generate sidecars for point clouds.
    - macOS / Linux: `conda install -c conda-forge pdal` (the tool currently looks for `/opt/conda/envs/pdal/bin/pdal`)
- **AWS CLI** — required only if you point `--directory` (or a client sidecar/schema) at an `s3://` URI. Install from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html and see the *Authenticating with AWS* section below.

### Python package

Requires Python 3.10+. In terminal or command prompt, open this folder and run the following command.

```bash
pip install -e ".[dev]"
```
This will install both `am-tools` as well as all requirements for `am-tools`.

## Authenticating with AWS

When `--directory` (or `--client-sidecar`, `--client-schema`) is an `s3://` URI, `am-tools` uses your local AWS credentials via `boto3`. The recommended setup is AWS SSO through the AWS CLI:

1. Install the AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
2. Configure an SSO profile and sign in. For a walkthrough of the one-time `~/.aws/config` setup, see: https://dev.to/slsbytheodo/understand-the-aws-sso-login-configuration-4am7
3. Run `aws sso login --profile <your-profile>` whenever your session expires.
4. Set `AWS_PROFILE=<your-profile>` (or rely on the `default` profile) before running `am-tools`.

If credentials are missing or insufficient, `am-tools` prints a bright-red help block after the stack trace telling you exactly what to do next — whether that's installing the AWS CLI, running `aws sso login`, or asking your administrator to grant a specific S3 IAM action.

## Usage
To use the library, open this folder in your terminal. The base command is:

```bash
am-tools --help
```

> Note: `amtools` without the dash also works.

### `am-tools sidecar`
This command is used to generate a valid sidecar for your data. It is important to note that data will only need a sidecar if the files are missing metadata for some reason but you are still welcome to use this command to validate your data. If the data is missing metadata, warnings will show when runing this command.

The primary use of `am-tools sidecar generate` is to format a client-provided sidecar into the spec that Atomic Flow needs for processing. When you run it, it will prompt you for some information about your sidecar.

- Directory
    - Where the files are that you want to build a sidecar for. This file can be local or on s3 or some other web object store. For S3, pass the whole s3 path. If the folder has subfolders, all files in the subfolders will also have the sidecar generated for them.
- Data Type
    - What type of data are you building a sidecar for? Different data types have different required columns and this handles that. Note: You can only generate a sidecar for one data type at at time. If you have multiple image types in the same folder (oriented images from a drone and spherical images in the same folder for example), the sidecar generation will fail.
- Output Filename
    - What would you like the sidecar to be named?
- Client-Supplied Sidecar
    - If you have your own sidecar data you want to merge in, link to it here. Without providing this information, the generated sidecar will just be made of the metadata already in the files. Sidecars made without client-supplied sidecars merged in are not required for processing because Atomic Flow uses very similar logic to get the file metadata automatically.
    - You can link to a single CSV, or to a **directory**. When you give a directory, every CSV in a subfolder *below* that directory is merged into a single sidecar. The directory itself is deliberately **not** scanned — that's where the generated sidecar gets written, so scanning it would pick up the output. This lets you keep one client CSV per subfolder (for example, one per flight or per delivery) and merge them all in one run. All of the CSVs are assumed to share the same format/schema; if one of them has a different number of columns, the run stops and tells you exactly which file is the odd one out.
- Client schema
    - If the client sidecar has no header or needs columns to be renamed, you can provide the schema to do that here. See below for more information on how to format the schema. By default it will look in the `./schemas` folder but you can provide a link to any file or to s3.
- Spatial Reference
    - If you know the spatial reference of your data and it isnt `EPSG:4326` (WGS:84), input it here to reproject your data to `EPSG:4326`. This is especially useful for pointclouds which will fail if we don't know the spatial reference.
- Include fields?
    - By default it only includes sidecar for required fields. However, you can manually have it include all of the fields if you want to see a complete list of metadata fields for your files. This is mostly used for debugging.
- Verbosity
    - How much do you want it to talk while doing the work?

After completing the interactive portion, it will show you in blue text a full command you can copy and paste to run it again without having to do the interactive wizard section.

After the sidecar has been generated, it will tell you where it put the file and it lints the file automatically. Speaking of linting, if you want to lint things manually, there is a command for that.

#### Reformatting a local sidecar
You can use `sidecar generate` to reformat a sidecar you already have locally into the Atomic Flow schema. The command scans the directory of data files and merges your CSV into the metadata it extracts, so point `--directory` at the folder that holds the actual files and `--client-sidecar` at your CSV. All flags below also accept `s3://`, `gs://`, and `az://` URIs.

**Reformat a local sidecar to the AM schema.** Provide a `--client-schema` to rename your columns into their canonical names (see *Sidecar Schemas* below):

```bash
am-tools sidecar generate \
  --directory ./data \
  --datatype oriented_image \
  --client-sidecar ./my_sidecar.csv \
  --client-schema ./schemas/my_schema.json
```

**Reformat and reproject from an input EPSG.** If your sidecar coordinates are not already in `EPSG:4326` (WGS84), add `--spatial-reference`. For images/videos the lat/lon (and altitude) are treated as X/Y/Z in that CRS and reprojected to `EPSG:4326`; for point clouds the value is recorded in a `spatial_reference` column instead:

```bash
am-tools sidecar generate \
  --directory ./data \
  --datatype oriented_image \
  --client-sidecar ./my_sidecar.csv \
  --spatial-reference EPSG:32612
```

**Reformat and merge two local sidecars.** When `--client-sidecar` is a **directory** instead of a single file, every CSV in a subfolder *below* that directory is merged into one sidecar (the directory itself is not scanned, since that is where the generated sidecar gets written). Put each CSV in its own subfolder and point `--client-sidecar` at the parent — all of them must share the same column schema:

```bash
# sidecars/flight-a/a.csv  and  sidecars/flight-b/b.csv  are merged
am-tools sidecar generate \
  --directory ./data \
  --datatype oriented_image \
  --client-sidecar ./sidecars
```

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
The command `am-tools lint sidecar` can lint both the client provided sidecar as well as the generated sidecar that `am-tools sidecar generate` builds. The first thing the linter wil ask you after a link to the sidecar is if the sidecar is generated or not. For linting a client provided sidecar, it will skip certain checks because it assumes that the files will have metadata that will be combined with the client sidecar.

For linting the generated sidecar, it will ask for data type. This is used to make sure it has all of the required columns for that data type. There are some basic checks for required fields to make sure the values make sense. If any fields fail this test, the script will tell you possible fixes.

## Project layout

```
src/atomic_tools/
├── cli.py              # root Typer app
├── client_sidecar.py   # load + clean + merge a client-supplied sidecar CSV
├── commands/           # one module per subcommand group (lint, sidecar)
├── io/                 # storage backend abstraction (local + s3/gs/az)
├── utils/              # helpers: exiftool/pdal extractors, object-store I/O
└── validators/         # schema/sidecar/value validators + lint reports
schemas/                # example client schema JSONs
example-fake-data/      # sample images + sidecars for trying the tool
tests/                  # pytest suite
```

## In Depth Spec for Sidecars
`am-tools sidecar generate` produces a sidecar that follows this spec automatically. The details below are mainly useful if you're hand-writing a client sidecar or debugging a generated one.

### Layout

1. **The first column must be `Filename`.** Each row identifies the file it applies to by basename (e.g. `IMG_001.jpg`). If two files in the batch share a basename, use a path with enough parent directories to disambiguate (`subfolderA/IMG_001.jpg` vs `subfolderB/IMG_001.jpg`).
2. **The first row after the header must be `DEFAULT`.** Values placed in this row apply to every file that doesn't have its own row or has an empty cell for that column. Resolution order, highest to lowest: value extracted from the file's own metadata → per-file row value → `DEFAULT` row value.
3. **Column names are case-insensitive but otherwise exact.** Whitespace, punctuation, and spelling all matter (`GPS Latitude` is not `GPSLatitude`).
4. **Cells may be left blank.** A blank cell means "fall back to the next source" per the resolution order above.

### Required columns by data type

Each data type requires one column from each required row below (the canonical name is shown; accepted aliases are listed beneath). Optional columns are used when present but never block processing:

| Data type | Required canonical columns | Optional columns |
|---|---|---|
| `ortho_image` | `Filename`, `GPSLatitude`, `GPSLongitude`, `GPSAltitude` | `CreateDate` |
| `oriented_image` | `Filename`, `GPSLatitude`, `GPSLongitude`, `GPSAltitude` | `CreateDate`, `Pitch`, `Heading`, `Roll` |
| `spherical_image` | `Filename`, `GPSLatitude`, `GPSLongitude`, `GPSAltitude`, `Pitch`, `Heading`, `Roll` | `CreateDate` |
| `point_cloud` | `Filename`, `bounds.minx`, `bounds.maxx`, `bounds.miny`, `bounds.maxy`, `bounds.minz`, `bounds.maxz`, `num_points`, `creation_year`, `creation_doy` | — |
| `video` | `Filename` | `CreateDate` |

Notes on optional columns:
- **Orientation** — If an `oriented_image` is missing `Pitch`, `Heading`, or `Roll`, it will appear in Lens without orientation but still processes successfully.
- **Dates** — `CreateDate` is optional: if omitted, Lens will try to find the acquisition date from the filename. Processing fails only if no date can be found in the metadata *or* the filename.

Common aliases that the linter accepts and rewrites to their canonical name:
- **Dates** — `CreateDate`, `DateTimeOriginal`, `ModifyDate`, `GPSDateStamp` are interchangeable.
- **Pitch** — `CameraPitch`, `CameraPitchDegree`, `GimbalPitchDegree`, `PosePitchDegrees`, `CameraOrientationNEDPitch`, `GPSIMUPitch`, `PitchAngle`.
- **Heading** — `Yaw`, `CameraYaw`, `CameraYawDegree`, `GimbalYawDegree`, `PoseHeadingDegrees`, `CameraOrientationNEDYaw`, `GPSIMUYaw`, `YawAngle`, `GPSImgDirection`, `imgDirection`.
- **Roll** — `CameraRoll`, `CameraRollDegree`, `GimbalRollDegree`, `PoseRollDegrees`, `CameraOrientationNEDRoll`, `GPSIMURoll`, `RollAngle`.

See [`src/atomic_tools/validators/required_fields.py`](src/atomic_tools/validators/required_fields.py) for the authoritative list.

### Value formats

- **GPSLatitude** — decimal in `[-90, 90]` (e.g. `51.0444`) or EXIF DMS (`51 deg 2' 40.92" N`).
- **GPSLongitude** — decimal in `[-180, 180]` or EXIF DMS.
- **GPSAltitude** — any finite number. A trailing unit such as `" m Above Sea Level"` is tolerated; the leading number is what's checked.
- **Dates** — ISO 8601 (`2024-06-15T10:30:00+00:00`, `2024-06-15 10:30:00`, or just `2024-06-15`) or EXIF (`2024:06:15 10:30:00`).
- **Pitch / Roll** — decimal degrees in `[-180, 180]`.
- **Heading** — decimal degrees in `[-360, 360]`.
- **`bounds.*`** — any finite number. `bounds.min<axis>` must be ≤ `bounds.max<axis>` for each of x/y/z.
- **`num_points`, `creation_year`, `creation_doy`** — finite numbers.

### Normalising a non-conforming client sidecar

If your CSV doesn't already match the spec, supply a schema JSON via `--client-schema`:
- Use `column_names` if your CSV has no header row.
- Use `column_name_mapping` to rename columns into their canonical names.

See the *Sidecar Schemas* section above for the schema format. Run `am-tools lint sidecar <your.csv> --schema <schema.json> --datatype <type>` to check it before generating.
