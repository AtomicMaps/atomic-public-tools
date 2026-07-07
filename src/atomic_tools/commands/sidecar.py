"""Generate a sidecar CSV by scanning a local or remote directory.

Originally `tasks/generate_sidecar_csv/generate_sidecar_csv.py`.

Requires the following external binaries at runtime (NOT pip-installable):

  * exiftool
  * pdal  (located at runtime by ``utils/extractors.find_pdal_bin`` — it
          searches PATH, the active conda env, any conda env named ``pdal``,
          and common install roots; override with the ``PDAL_BIN`` env var)
"""

import logging
import shlex
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple, NoReturn

import click
import questionary
import typer
from tqdm import tqdm

if TYPE_CHECKING:
    from atomic_tools.cli import VerbosityChoice

from atomic_tools.client_sidecar import (
    build_global_alias_map,
    load_and_clean_client_sidecar,
    merge_client_metadata,
)
from atomic_tools.io.storage import StorageBackend, from_directory
from atomic_tools.utils.aws_errors import find_auth_error, print_help_block
from atomic_tools.utils.coordinates import (
    WEB_MERCATOR_EPSG,
    can_transform_to_web_mercator,
    transform_center_to_web_mercator,
    transform_coordinates,
)
from atomic_tools.utils.extractors import (
    PdalNotFoundError,
    extract_exif_metadata,
    extract_pdal_crs,
    extract_pdal_metadata,
    find_pdal_bin,
    infer_date_from_filepath,
    spherical_signals_from_exif,
)
from atomic_tools.utils.utils import (
    DATA_TYPE_INFO,
    DataTypeEnum,
    DataTypeFilter,
    _split_path_components,
    has_value,
    infer_data_type,
    is_remote_uri,
)
from atomic_tools.validators.coco import IMAGE_DATA_TYPES
from atomic_tools.validators.required_fields import (
    ALL_SIDECAR_FIELD_GROUPS as _ALL_SIDECAR_FIELD_GROUPS,
)
from atomic_tools.validators.required_fields import (
    REQUIRED_SIDECAR_FIELD_GROUPS as _REQUIRED_SIDECAR_FIELD_GROUPS,
)
from atomic_tools.validators.sidecar import lint_sidecar_file
from atomic_tools.validators.values import parse_elevation, to_decimal_degree

logger = logging.getLogger(__name__)

# Image data types share the EXIF extractor and are the only types COCO label
# impact applies to — IMAGE_DATA_TYPES is the single source of truth (coco.py).
_VIDEO_DATA_TYPES = {DataTypeEnum.video}
_POINT_CLOUD_DATA_TYPES = {DataTypeEnum.point_cloud}

# Customer-facing per-point-cloud status column: did the effective CRS convert to
# Web Mercator? Populated by _add_crs_web_mercator_column.
_CRS_WEB_MERCATOR_COL = "crs_web_mercator_ok"


def _fail(message: str, exc: Exception) -> NoReturn:
    """Log a failure, surface AWS auth help if applicable, and exit non-zero."""
    # A missing pdal is a setup problem, not a bug — show the install
    # instructions plainly rather than burying them under a traceback.
    if isinstance(exc, PdalNotFoundError):
        logger.error(f"{message}: {exc}")
        raise typer.Exit(code=1) from None
    logger.exception(message)
    auth_err = find_auth_error(exc)
    if auth_err is not None:
        print_help_block(auth_err)
    raise typer.Exit(code=1) from None


def _list_local_schemas() -> list[Path]:
    """Return sorted ``*.json`` files from ``./schemas/`` if that dir exists."""
    schemas_dir = Path.cwd() / "schemas"
    if not schemas_dir.is_dir():
        return []
    return sorted(schemas_dir.glob("*.json"))


def _ask_directory() -> str:
    return (
        questionary.text(
            "Directory to scan:",
            instruction="(Local folder or object-store URI like s3://bucket/prefix)",
            validate=lambda v: bool(v.strip()) or "Required.",
        )
        .unsafe_ask()
        .strip()
    )


def _ask_data_type() -> DataTypeEnum:
    choice = questionary.select(
        "Data type of the input data:",
        choices=[v.value for v in DataTypeEnum if v in _REQUIRED_SIDECAR_FIELD_GROUPS],
    ).unsafe_ask()
    return DataTypeEnum(choice)


_AUTO_DETECT_CHOICE = "Auto-detect (recommended)"


def _ask_data_type_filter() -> DataTypeEnum | None:
    """Prompt for an optional data-type filter; default is auto-detect.

    Returns None for auto-detect (classify every file), else the chosen type.
    """
    choice = questionary.select(
        "Data type filter:",
        instruction=(
            "(Auto-detect classifies every file; choose a type only to restrict "
            "the scan to it)"
        ),
        choices=[_AUTO_DETECT_CHOICE, *(v.value for v in DataTypeFilter)],
        default=_AUTO_DETECT_CHOICE,
    ).unsafe_ask()
    if choice == _AUTO_DETECT_CHOICE:
        return None
    return DataTypeEnum(choice)


def _ask_ignore_missing_orientation() -> bool:
    return questionary.confirm(
        "Ignore missing orientation data (Pitch/Heading/Roll)?",
        instruction=(
            "(No — the default — treats a missing orientation field as an error; "
            "Yes downgrades it to a warning so the images still process, "
            "appearing in Lens without orientation)"
        ),
        default=False,
    ).unsafe_ask()


def _ask_output_filename() -> str:
    return questionary.text(
        "Output filename:",
        default="sidecar.csv",
        instruction="(Press Enter to keep the default)",
    ).unsafe_ask()


def _ask_full() -> bool:
    return questionary.confirm(
        "Include every metadata field exiftool/pdal extracted? "
        "(No keeps only the canonical fields Flow needs)",
        default=False,
    ).unsafe_ask()


def _ask_local_copy() -> bool:
    return questionary.confirm(
        "Also save a local copy of the sidecar in this directory "
        "(so you can inspect it before submitting)?",
        default=True,
    ).unsafe_ask()


_VERBOSITY_DEFAULT = "Default (warnings only)"
_VERBOSITY_VERBOSE = "Verbose (info)"
_VERBOSITY_SILENT = "Silent (errors only)"


def _ask_verbosity() -> "VerbosityChoice":
    selection = questionary.select(
        "Logging verbosity:",
        choices=[_VERBOSITY_DEFAULT, _VERBOSITY_VERBOSE, _VERBOSITY_SILENT],
        default=_VERBOSITY_DEFAULT,
    ).unsafe_ask()
    if selection == _VERBOSITY_VERBOSE:
        return "verbose"
    if selection == _VERBOSITY_SILENT:
        return "silent"
    return "default"


def _ask_client_sidecar() -> str | None:
    answer = questionary.text(
        "Optional client-supplied sidecar CSV to merge in:",
        instruction=(
            "(Local path or s3:// URI to a CSV, or a directory whose "
            "subfolders hold the CSVs to merge; press Enter to skip)"
        ),
    ).unsafe_ask()
    answer = answer.strip()
    return answer or None


def _ask_coco() -> str | None:
    answer = questionary.text(
        "Optional COCO label file to assess label impact:",
        instruction=(
            "(Local path or s3:// URI to a COCO .json, or a directory containing "
            "one; press Enter to skip)"
        ),
    ).unsafe_ask()
    answer = answer.strip()
    return answer or None


_SCHEMA_CUSTOM = "Custom local path…"
_SCHEMA_S3 = "S3 link (s3:// URI)…"
_SCHEMA_SKIP = "Skip"


def _validate_schema_input(v: str) -> bool | str:
    v = v.strip()
    if not v:
        return "Required."
    if is_remote_uri(v):
        return True
    return Path(v).expanduser().is_file() or "File not found (and not a remote URI)."


def _validate_remote_uri(v: str) -> bool | str:
    v = v.strip()
    if not v:
        return "Required."
    return is_remote_uri(v) or "Expected an s3:// (or gs:// / az://) URI."


def ask_schema_uri(prompt: str = "Path or URI to schema JSON:") -> str:
    """Prompt for a schema path or URI; resolve local paths, pass URIs through."""
    answer = (
        questionary.text(
            prompt,
            instruction="(Local path or s3://… / gs://… / az://… URI)",
            validate=_validate_schema_input,
        )
        .unsafe_ask()
        .strip()
    )
    if is_remote_uri(answer):
        return answer
    return str(Path(answer).expanduser().resolve())


def ask_remote_schema_uri(prompt: str = "S3 link to schema JSON:") -> str:
    """Prompt for a remote (s3/gs/az) schema URI, passed through as-is."""
    return (
        questionary.text(
            prompt,
            instruction="(s3://… / gs://… / az://… URI)",
            validate=_validate_remote_uri,
        )
        .unsafe_ask()
        .strip()
    )


def _ask_client_schema() -> str | None:
    schemas = _list_local_schemas()
    choices = [str(p.name) for p in schemas] + [
        _SCHEMA_CUSTOM,
        _SCHEMA_S3,
        _SCHEMA_SKIP,
    ]
    selection = questionary.select(
        "Optional client schema (normalises a client-supplied sidecar):",
        choices=choices,
        default=_SCHEMA_SKIP,
    ).unsafe_ask()
    if selection == _SCHEMA_SKIP:
        return None
    if selection == _SCHEMA_CUSTOM:
        return ask_schema_uri()
    if selection == _SCHEMA_S3:
        return ask_remote_schema_uri()
    return str(next(p for p in schemas if p.name == selection))


def _ask_spatial_reference() -> str | None:
    answer = questionary.text(
        "Optional spatial reference (CRS) of the source coordinates:",
        instruction=(
            "(e.g. 'EPSG:32612' or '32612'. For images/videos lat/lon/altitude "
            "are reprojected to EPSG:4326; for point clouds it's recorded as a "
            "column. Press Enter to skip)"
        ),
    ).unsafe_ask()
    answer = answer.strip()
    return answer or None


def _disambiguate_filenames(keys: list[str]) -> dict[str, str]:
    """Return a per-key display label, extended with the minimum number of
    parent directories needed to be unique across ``keys``.
    """
    parts_per_key: dict[str, tuple[str, ...]] = {
        key: _split_path_components(key) for key in keys
    }
    depths: dict[str, int] = {key: 1 for key in keys}

    while True:
        labels = {
            key: "/".join(parts[-depths[key] :]) for key, parts in parts_per_key.items()
        }
        counts: dict[str, int] = defaultdict(int)
        for label in labels.values():
            counts[label] += 1
        collisions = {label for label, n in counts.items() if n > 1}
        if not collisions:
            return labels
        bumped = False
        for key, label in labels.items():
            if label in collisions and depths[key] < len(parts_per_key[key]):
                depths[key] += 1
                bumped = True
        if not bumped:
            return labels


def _split_gps_position(meta: dict) -> dict:
    """Split a ``GPSPosition`` value into ``GPSLatitude``/``GPSLongitude``.

    exiftool emits ``GPSPosition`` as ``"<lat>, <lon>"`` (DMS or decimal),
    or as a 2-element list when the ``-n`` flag is in play. When a file or
    client sidecar provides ``GPSPosition``, treat it as authoritative and
    overwrite ``GPSLatitude``/``GPSLongitude`` with the split halves.
    Returns `meta` unchanged if ``GPSPosition`` is absent or unparseable.
    """
    raw = meta.get("GPSPosition")
    if raw is None:
        return meta
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            return meta
        lat, lon = (str(part).strip() for part in raw)
    else:
        text = str(raw)
        if "," not in text:
            return meta
        lat, lon = (part.strip() for part in text.split(",", 1))
    if not lat or not lon:
        return meta
    out = {k: v for k, v in meta.items() if k != "GPSPosition"}
    out["GPSLatitude"] = lat
    out["GPSLongitude"] = lon
    return out


_DATE_ALIASES = {"CreateDate", "DateTimeOriginal", "ModifyDate", "GPSDateStamp"}


def _date_group(required_field_groups: list[list[str]]) -> list[str] | None:
    for group in required_field_groups:
        if set(group) & _DATE_ALIASES:
            return group
    return None


def _fill_missing_dates_from_filepath(
    file_metadata: list[tuple[str, dict]],
    required_field_groups: list[list[str]],
    display_to_key: dict[str, str],
) -> None:
    """For each file with no date field after merge, infer one from its path."""
    date_group = _date_group(required_field_groups)
    if date_group is None:
        return
    canonical = date_group[0]
    for display_label, meta in file_metadata:
        if any(has_value(meta.get(field)) for field in date_group):
            continue
        original = display_to_key.get(display_label, display_label)
        lat = to_decimal_degree(meta.get("GPSLatitude"))
        lon = to_decimal_degree(meta.get("GPSLongitude"))
        inferred = infer_date_from_filepath(original, lat, lon)
        if inferred is None:
            continue
        formatted = inferred.isoformat()
        meta[canonical] = formatted
        logger.warning(
            f"Inferred {canonical} for {display_label!r} from file path: "
            f"{formatted}. No date was provided by EXIF or the client sidecar."
        )


def _warn_missing_required_fields(
    file_metadata: list[tuple[str, dict]],
    required_field_groups: list[list[str]],
    type_label: str | None = None,
) -> None:
    """Emit a loud warning when files lack required fields after the merge.

    A group is "satisfied" for a file if any field in the group is present
    (non-empty) in that file's metadata. For each unsatisfied group we log a
    structured WARNING and also print a bright-red message to stderr so the
    operator notices that the client sidecar needs to be updated. ``type_label``
    prefixes the messages with the data type when a scan detected more than one.
    """
    if not file_metadata or not required_field_groups:
        return

    prefix = f"[{type_label}] " if type_label else ""

    missing_by_group: list[tuple[str, list[str], list[str]]] = []
    for group in required_field_groups:
        canonical = group[0]
        missing_files = [
            filename
            for filename, meta in file_metadata
            if not any(has_value(meta.get(field)) for field in group)
        ]
        if missing_files:
            missing_by_group.append((canonical, list(group), missing_files))

    if not missing_by_group:
        return

    total = len(file_metadata)
    logger.warning(
        f"{prefix}Missing required metadata after client sidecar merge — "
        "see details below."
    )

    header = (
        f"{prefix}MISSING REQUIRED METADATA: {len(missing_by_group)} required "
        f"field(s) are not satisfied for every file. Update the client sidecar "
        f"to provide values for the listed files."
    )
    click.secho(header, fg="bright_red", bold=True, err=True, color=True)

    for canonical, group, missing_files in missing_by_group:
        accepted = ", ".join(group)
        sample = missing_files if len(missing_files) <= 5 else missing_files[:5] + ["…"]
        line = (
            f"  [{canonical}] missing on {len(missing_files)}/{total} file(s): "
            f"{sample} (accepted aliases: {accepted})"
        )
        click.secho(line, fg="bright_red", err=True, color=True)


def _canonicalize_keys(meta: dict, alias_to_canonical: dict[str, str]) -> dict:
    """Rename alias keys in `meta` to their canonical names. If the canonical
    is already present, the alias is dropped; among multiple aliases for the
    same canonical, the first-encountered one wins.
    """
    out: dict = {
        key: value for key, value in meta.items() if key not in alias_to_canonical
    }
    for key, value in meta.items():
        if key in alias_to_canonical:
            out.setdefault(alias_to_canonical[key], value)
    return out


def _union_field_groups(
    field_groups_by_type: dict[str, list[list[str]]],
) -> list[list[str]]:
    """Order-stable dedup of every detected type's field groups into one list."""
    union: list[list[str]] = []
    seen_ids: set[tuple[str, ...]] = set()
    for groups in field_groups_by_type.values():
        for group in groups:
            gid = tuple(group)
            if gid not in seen_ids:
                seen_ids.add(gid)
                union.append(group)
    return union


def build_sidecar_df(
    file_metadata: list[tuple[str, dict]],
    field_groups_by_type: dict[str, list[list[str]]],
    types_by_label: dict[str, str],
    full: bool = False,
):
    """Assemble the sidecar DataFrame.

    Layout:
      - Row 0  : DEFAULT row — Filename="DEFAULT", DataType="", other columns empty
      - Row 1+ : one row per file with extracted metadata values, sorted by Filename
      - Col 0  : "Filename" (disambiguated basename of the source key/path)
      - Col 1  : "DataType" (the detected type per file; "" on DEFAULT)

    ``field_groups_by_type`` maps each *detected* data type (value string) to its
    field groups; ``types_by_label`` maps each file's display label to its type.
    Alias canonicalization and (when ``full`` is False) column selection use the
    union of all detected types' groups. Blank required columns are prepended
    **per type**: a group's canonical is prepended iff some row *of that type*
    lacks every field in the group — so a mixed scan never pollutes image rows
    with blank point-cloud columns or vice versa.
    """
    import pandas as pd

    if not file_metadata:
        return pd.DataFrame(columns=["Filename", "DataType"])

    union_groups = _union_field_groups(field_groups_by_type)
    alias_to_canonical = build_global_alias_map(union_groups)
    file_metadata = [
        (filename, _canonicalize_keys(_split_gps_position(meta), alias_to_canonical))
        for filename, meta in file_metadata
    ]

    all_cols: list[str] = []
    seen: set[str] = set()
    for _, meta in file_metadata:
        for col in meta:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    if not full:
        allowed = {field for group in union_groups for field in group}
        all_cols = [c for c in all_cols if c in allowed]

    # Per-type blank-column prepending: only prepend a group's canonical when a
    # row *of that type* is missing every field in the group.
    prepend_cols: list[str] = []
    for type_value, groups in field_groups_by_type.items():
        rows_of_type = [
            meta for label, meta in file_metadata if types_by_label.get(label) == type_value
        ]
        if not rows_of_type:
            continue
        for group in groups:
            canonical = group[0]
            all_covered = all(
                any(field in meta for field in group) for meta in rows_of_type
            )
            if not all_covered and canonical not in prepend_cols:
                prepend_cols.append(canonical)

    all_cols = prepend_cols + [c for c in all_cols if c not in prepend_cols]
    columns = ["Filename", "DataType", *all_cols]

    rows = [
        {
            "Filename": filename,
            "DataType": types_by_label.get(filename, ""),
            **{col: meta.get(col, "") for col in all_cols},
        }
        for filename, meta in file_metadata
    ]
    df = pd.DataFrame(rows, columns=columns)
    # DEFAULT is always first (prepended below); the rest sort by DataType then
    # Filename so rows of the same type group together, alphabetically within.
    df = df.sort_values(by=["DataType", "Filename"], kind="stable", ignore_index=True)

    default_row = pd.DataFrame(
        [{"Filename": "DEFAULT", "DataType": "", **{col: "" for col in all_cols}}]
    )
    return pd.concat([default_row, df], ignore_index=True)


def _add_spatial_reference_column(df, spatial_reference: str) -> None:
    """Add a 'fallback_srs' column, populated only on the DEFAULT row."""
    df["fallback_srs"] = ""
    df.loc[df["Filename"] == "DEFAULT", "fallback_srs"] = spatial_reference


def _add_file_srs_column(df, file_metadata: list[tuple[str, dict]]) -> None:
    """Add a 'file_srs' column holding each file's PDAL-read header CRS.

    Blank for files whose header carried no CRS (those rely on the DEFAULT row's
    'fallback_srs') and blank on the DEFAULT row itself.
    """
    crs_by_label = {label: (extract_pdal_crs(meta) or "") for label, meta in file_metadata}
    df["file_srs"] = df["Filename"].map(lambda name: crs_by_label.get(name, ""))


def _reproject_dataframe(df, in_srs: str, only_labels: set[str] | None = None) -> None:
    """Reproject GPSLongitude/GPSLatitude (and GPSAltitude as Z) from `in_srs`
    to EPSG:4326 in place, skipping the DEFAULT row and unparseable coordinates.

    When ``only_labels`` is given, only rows whose Filename is in that set are
    reprojected — used in mixed scans to reproject image/video rows while
    leaving point-cloud rows (which carry a CRS per-file) untouched.
    """
    for idx in df.index:
        if df.at[idx, "Filename"] == "DEFAULT":
            continue
        if only_labels is not None and df.at[idx, "Filename"] not in only_labels:
            continue
        lon = to_decimal_degree(df.at[idx, "GPSLongitude"])
        lat = to_decimal_degree(df.at[idx, "GPSLatitude"])
        if lon is None or lat is None:
            continue  # leave blank/unparseable rows untouched
        z = (
            parse_elevation(df.at[idx, "GPSAltitude"])
            if "GPSAltitude" in df.columns
            else None
        )
        if z is None:
            nx, ny = transform_coordinates(lon, lat, in_srs, 4326)
        else:
            nx, ny, nz = transform_coordinates(lon, lat, in_srs, 4326, z=z)
            df.at[idx, "GPSAltitude"] = str(nz)
        df.at[idx, "GPSLongitude"] = str(nx)
        df.at[idx, "GPSLatitude"] = str(ny)


def _bbox_center_from_meta(meta: dict) -> tuple[float, float, float | None] | None:
    """Return ``(cx, cy, cz)`` — the bbox-center midpoints — from PDAL metadata.

    ``cz`` is None when Z bounds are absent. Returns None if the X/Y bounds are
    missing or non-numeric, so the caller can fall back to a CRS-only check.
    """

    def midpoint(lo_key: str, hi_key: str) -> float | None:
        try:
            return (float(meta[lo_key]) + float(meta[hi_key])) / 2.0
        except (KeyError, TypeError, ValueError):
            return None

    cx = midpoint("bounds.minx", "bounds.maxx")
    cy = midpoint("bounds.miny", "bounds.maxy")
    if cx is None or cy is None:
        return None
    return cx, cy, midpoint("bounds.minz", "bounds.maxz")


def _crs_reaches_web_mercator(meta: dict, crs: str) -> bool:
    """True if ``crs`` transforms to EPSG:3857 for this file.

    Prefers transforming the file's actual bounding-box center into EPSG:3857 (a
    stricter test than merely building a transformer — it also catches a CRS that
    maps real coordinates to non-finite values), falling back to a CRS-only check
    when bounds are unavailable.
    """
    center = _bbox_center_from_meta(meta)
    if center is not None:
        return transform_center_to_web_mercator(*center, crs) is not None
    return can_transform_to_web_mercator(crs)


def _add_crs_web_mercator_column(
    df,
    pc_metadata: list[tuple[str, dict]],
    spatial_reference: str | None,
) -> None:
    """Add a customer-facing ``crs_web_mercator_ok`` status column for point clouds.

    Flow renders point clouds in Web Mercator, so each one must resolve to a CRS
    pyproj can transform to EPSG:3857 — its header CRS (``file_srs``), else the
    ``--spatial-reference`` fallback (``fallback_srs``). For every point-cloud row
    this records ``"yes"`` when that effective CRS converts successfully and
    ``"no"`` when it is missing or cannot be converted (a row the customer needs
    to fix). Blank on non-point-cloud rows and the DEFAULT row. Rows marked
    ``"no"`` are also reported as lint errors, so the status is a convenience, not
    a substitute for the linter's blocking check.
    """
    status: dict[str, str] = {}
    for label, meta in pc_metadata:
        crs = extract_pdal_crs(meta) or spatial_reference
        status[label] = "yes" if crs and _crs_reaches_web_mercator(meta, crs) else "no"
    df[_CRS_WEB_MERCATOR_COL] = df["Filename"].map(lambda name: status.get(name, ""))


def _crs_failures(df) -> list[str]:
    """Filenames of point-cloud rows whose CRS could not convert to Web Mercator.

    Reads the ``crs_web_mercator_ok`` column (present only when the scan found
    point clouds); a row is a failure when that flag is ``"no"``.
    """
    if _CRS_WEB_MERCATOR_COL not in df.columns:
        return []
    flag = df[_CRS_WEB_MERCATOR_COL].astype(str).str.strip()
    return df.loc[flag == "no", "Filename"].tolist()


def _raise_crs_failures_loudly(crs_failures: list[str]) -> None:
    """Print an unmissable bright-red error for point clouds that fail CRS conversion.

    The linter already records these as errors (so the run exits non-zero); this
    additionally surfaces them as a loud banner at the very end of the run, where
    they won't be lost among the rest of the lint report. No-op when nothing failed.
    """
    if not crs_failures:
        return
    sample = crs_failures if len(crs_failures) <= 10 else crs_failures[:10] + ["…"]
    click.secho(
        f"\nERROR: {len(crs_failures)} point cloud(s) could not be converted to "
        f"Web Mercator ({WEB_MERCATOR_EPSG}) — their CRS is missing or invalid, so "
        f"Flow cannot ingest them: {sample}",
        fg="bright_red",
        bold=True,
        err=True,
        color=True,
    )
    click.secho(
        f"  Fix the CRS in the file(s), or re-run with a valid --spatial-reference. "
        f"See the '{_CRS_WEB_MERCATOR_COL}' column (rows marked 'no').",
        fg="bright_red",
        err=True,
        color=True,
    )


def _union_include_extensions() -> list[str]:
    """Union (order-stable) of every data type's ``include`` extensions.

    Used for auto-detect listing: we list everything and let ``infer_data_type``
    (which applies per-type excludes itself) do the classification. Vector is
    included so vector files can be counted and warned about.
    """
    union: list[str] = []
    for info in DATA_TYPE_INFO.values():
        for ext in info.get("include") or []:
            if ext not in union:
                union.append(ext)
    return union


# Camera/software-generated derivative files (previews, thumbnails, annotation
# sidecars) that are never source data. In auto-detect these fail classification
# — they match a data type's ``exclude`` patterns — but they are expected noise,
# not "unknown types", so they are skipped silently rather than warned about.
_IGNORED_DERIVATIVE_SUFFIXES = (
    "previewimage.jpg",
    "thumbnailimage.jpg",
    "_thumbnail.jpg",
    "_thumbnail.jpeg",
    "_thumbnail.png",
    "annotation.json",
)


def _is_ignorable_derivative(key: str) -> bool:
    """True for preview/thumbnail/derivative files that should be skipped quietly.

    Matches the known derivative suffixes above — the exact patterns a data type's
    ``exclude`` list rejects, which are what make ``infer_data_type`` return None
    for these files in the first place.
    """
    return Path(key).name.lower().endswith(_IGNORED_DERIVATIVE_SUFFIXES)


def _classify_keys(
    keys: list[str],
) -> tuple[dict[str, DataTypeEnum], list[str], list[str]]:
    """Filename-only classification of listed keys for auto-detect mode.

    Returns ``(types_by_key, unclassified, vector_keys)``. ``types_by_key`` holds
    only supported (non-vector, classified) keys. Preview/thumbnail derivatives
    are dropped silently (see ``_is_ignorable_derivative``) — they never reach
    ``unclassified``, so they don't trigger the "no known data type" warning.
    """
    types_by_key: dict[str, DataTypeEnum] = {}
    unclassified: list[str] = []
    vector_keys: list[str] = []
    ignored: list[str] = []
    for key in keys:
        inferred = infer_data_type(key)
        if inferred is None:
            (ignored if _is_ignorable_derivative(key) else unclassified).append(key)
        elif inferred == DataTypeEnum.vector.value:
            vector_keys.append(key)
        else:
            types_by_key[key] = DataTypeEnum(inferred)
    if ignored:
        logger.debug(
            f"Silently skipped {len(ignored)} preview/thumbnail derivative file(s)."
        )
    return types_by_key, unclassified, vector_keys


def _warn_skipped(keys: list[str], message: str) -> None:
    sample = keys[:5]
    suffix = f" (+{len(keys) - 5} more)" if len(keys) > 5 else ""
    logger.warning(f"{message}: {sample}{suffix}")


def _warn_skipped_vectors(vector_keys: list[str]) -> None:
    """Report vector files skipped during the scan (called once, at the end).

    Vector sidecar generation isn't supported, so vector files are silently
    passed over while scanning; this surfaces the skip after the run so it isn't
    lost among the per-file progress output.
    """
    if not vector_keys:
        return
    sample = vector_keys[:5]
    suffix = f" (+{len(vector_keys) - 5} more)" if len(vector_keys) > 5 else ""
    click.secho(
        f"\nNote: {len(vector_keys)} vector file(s) were skipped — vector sidecar "
        f"generation is not supported: {sample}{suffix}",
        fg="yellow",
        err=True,
    )


_TIFF_SUFFIXES = {".tif", ".tiff"}


def _drop_preview_tifs(keys: list[str]) -> list[str]:
    """Drop ``.tif`` files that are generated previews of another file.

    Some capture pipelines emit a ``.tif`` preview alongside the real asset —
    e.g. ``R0010013.tif`` next to ``R0010013.JPG``. These are not unknown
    filetypes, just rasterized previews, and must not become sidecar rows. A
    ``.tif`` is treated as a preview when another (non-tiff) file in the *same
    folder* shares its basename stem. Standalone ortho ``.tif`` files (no such
    companion) are kept.
    """
    by_stem: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        p = Path(key)
        by_stem[(str(p.parent), p.stem.lower())].append(key)

    kept: list[str] = []
    dropped: list[str] = []
    for key in keys:
        p = Path(key)
        if p.suffix.lower() in _TIFF_SUFFIXES and any(
            other != key and Path(other).suffix.lower() not in _TIFF_SUFFIXES
            for other in by_stem[(str(p.parent), p.stem.lower())]
        ):
            dropped.append(key)
        else:
            kept.append(key)

    if dropped:
        sample = dropped[:5]
        suffix = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
        logger.info(
            f"Ignored {len(dropped)} preview .tif file(s) that share a name with "
            f"another file in the same folder: {sample}{suffix}"
        )
    return kept


def _drop_all_empty_columns(df, keep: set[str] | None = None) -> None:
    """Drop columns whose every value is blank, in place.

    ``Filename`` and ``DataType`` are always kept, as is any column in ``keep``
    (the required fields for the detected types — a blank required column is left
    in place as a checklist of what still needs filling). A non-protected column
    survives only if some cell (including the DEFAULT row) holds a value — so
    ``fallback_srs`` (set only on DEFAULT) stays, while columns no file populated
    are removed entirely.
    """
    protected = {"Filename", "DataType"} | (keep or set())
    empty_cols = [
        col
        for col in df.columns
        if col not in protected and not df[col].map(has_value).any()
    ]
    if empty_cols:
        logger.info(
            f"Dropping {len(empty_cols)} non-required column(s) with no values "
            f"in any row: {empty_cols}"
        )
        df.drop(columns=empty_cols, inplace=True)


def _move_blank_columns_to_end(df):
    """Return ``df`` with every all-blank column moved to the right edge.

    After ``_drop_all_empty_columns`` the only all-blank columns left are the
    protected required fields; grouping them at the end keeps the populated
    columns (``Filename``/``DataType`` first, then real data) together on the
    left and pushes the empty checklist columns out of the way.
    """
    blank = [col for col in df.columns if not df[col].map(has_value).any()]
    if not blank:
        return df
    non_blank = [col for col in df.columns if col not in blank]
    logger.info(f"Moving {len(blank)} all-blank column(s) to the end: {blank}")
    return df[non_blank + blank]


def _build_sidecar(
    directory: str,
    data_type: DataTypeEnum | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool = False,
    spatial_reference: str | None = None,
) -> tuple["object", StorageBackend, set[DataTypeEnum], list[str]]:
    """Scan a directory, extract metadata, and assemble the sidecar DataFrame.

    This is the shared core of both ``sidecar`` and ``validate``: it
    does everything up to (but not including) writing the CSV anywhere. Returns
    ``(df, backend, detected_types, skipped_vector_keys)`` so callers can persist
    the sidecar (``generate``) or lint it from a throwaway temp file
    (``validate``), log what was detected, and report skipped vector files at the
    end of the run rather than mid-scan.

    ``data_type`` is an optional filter/override:

    * ``None`` (auto): every file is classified with ``infer_data_type`` — the
      same code the backend uses. Ambiguous images are refined oriented↔spherical
      from EXIF signals. Unclassified and vector files are skipped with a warning.
    * an explicit type: byte-identical to the pre-auto behaviour — single-type
      include/exclude listing, no spherical refinement (a client passing
      ``--datatype spherical_image`` must not have panoramas reclassified).
    """
    if spatial_reference:
        # Fail fast on an unparseable CRS before doing any extraction work.
        from pyproj import CRS
        from pyproj.exceptions import CRSError

        try:
            CRS(spatial_reference)
        except CRSError:
            raise ValueError(
                f"--spatial-reference value {spatial_reference!r} is not a valid "
                "spatial reference (try e.g. 'EPSG:32612' or '32612')."
            ) from None

    backend = from_directory(directory)

    if data_type is not None:
        # --- Explicit override: single-type listing, no refinement. ---
        if data_type == DataTypeEnum.vector:
            raise NotImplementedError(
                "Vector data type is not yet supported for sidecar CSV generation."
            )
        if data_type not in DATA_TYPE_INFO:
            raise ValueError(
                f"Unsupported data_type '{data_type}'. "
                f"Valid values: {list(DATA_TYPE_INFO.keys())}"
            )
        info = DATA_TYPE_INFO[data_type]
        keys = backend.list_keys(
            include=list(info.get("include") or []),
            exclude=list(info.get("exclude") or []),
        )
        if not keys:
            raise RuntimeError(
                f"No valid files found for '{data_type.value}' in {backend.display_root}"
            )
        types_by_key: dict[str, DataTypeEnum] = {key: data_type for key in keys}
        vector_keys: list[str] = []
    else:
        # --- Auto-detect: list everything, classify per file. ---
        keys = backend.list_keys(include=_union_include_extensions(), exclude=[])
        keys = _drop_preview_tifs(keys)
        types_by_key, unclassified, vector_keys = _classify_keys(keys)
        if unclassified:
            _warn_skipped(
                unclassified,
                f"{len(unclassified)} file(s) matched no known data type and were skipped",
            )
        # Vector files are unsupported, but we don't interrupt the scan to say so
        # — the skip is reported once at the end (see _run_generate / _validate).
        if not types_by_key:
            raise RuntimeError(f"No supported files found in {backend.display_root}")

    supported_keys = list(types_by_key)
    total = len(supported_keys)
    logger.info(f"Found {total} file(s) to process in {backend.display_root}.")

    display_labels = _disambiguate_filenames(supported_keys)
    display_to_key = {v: k for k, v in display_labels.items()}
    n_disambiguated = sum(
        1 for k, v in display_labels.items() if v != Path(k).name
    )
    if n_disambiguated:
        logger.info(
            f"Disambiguated {n_disambiguated} file(s) whose basenames collided "
            f"by prepending parent directories."
        )

    # For point clouds, confirm pdal is available up front so we fail fast with
    # install instructions instead of logging the same "pdal not found" warning
    # once per file as we walk the directory.
    if any(t == DataTypeEnum.point_cloud for t in types_by_key.values()):
        find_pdal_bin()

    file_metadata: list[tuple[str, dict]] = []
    empty_count = 0

    with tqdm(
        supported_keys,
        desc="Extracting metadata",
        unit="file",
        file=sys.stderr,
        disable=None,
    ) as progress_keys:
        for i, key in enumerate(progress_keys, 1):
            progress_keys.set_postfix_str(Path(key).name if key else "")
            display_label = display_labels[key]
            key_type = types_by_key[key]
            is_pc = key_type == DataTypeEnum.point_cloud
            tool = "PDAL" if is_pc else "EXIF"
            logger.info(f"[{i}/{total}] Extracting {tool}: {key}")
            with backend.open_local(key) as local_path:
                if is_pc:
                    meta = extract_pdal_metadata(local_path, filename=display_label)
                else:
                    meta = extract_exif_metadata(local_path, filename=display_label)
                    # Auto mode only: refine ambiguous images oriented↔spherical
                    # from EXIF signals. The filename already chose the (shared)
                    # EXIF extractor, so this never changes which extractor ran.
                    if data_type is None and key_type in IMAGE_DATA_TYPES:
                        refined = infer_data_type(
                            key, **spherical_signals_from_exif(meta)
                        )
                        if refined is not None:
                            types_by_key[key] = DataTypeEnum(refined)
            if not meta:
                empty_count += 1
                logger.warning(f"No {tool} metadata returned for {key}")
            file_metadata.append((display_label, meta))

    if empty_count:
        logger.warning(
            f"{empty_count}/{total} file(s) produced empty metadata — "
            "those rows will have blank fields in the sidecar."
        )

    detected_types = set(types_by_key.values())
    counts = Counter(t.value for t in types_by_key.values())
    detected_summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
    logger.info(f"Detected: {detected_summary}")

    # Per-file display-label → type-value string, reflecting any refinement.
    types_by_label: dict[str, str] = {
        display_labels[k]: t.value for k, t in types_by_key.items()
    }

    def _labels_for(*want: DataTypeEnum) -> set[str]:
        want_set = set(want)
        return {display_labels[k] for k, t in types_by_key.items() if t in want_set}

    pc_labels = _labels_for(DataTypeEnum.point_cloud)
    pc_metadata = [(label, meta) for label, meta in file_metadata if label in pc_labels]

    if client_sidecar:
        client_url = client_sidecar.strip()
        if not client_url:
            raise ValueError("--client-sidecar value cannot be empty.")
        logger.info(f"Merging client sidecar from {client_url}")
        client_df = load_and_clean_client_sidecar(
            url=client_url,
            schema_path=client_schema,
            required_field_groups=_ALL_SIDECAR_FIELD_GROUPS,
        )
        # A client-supplied DataType column would clobber the detected value on
        # merge; the detection is authoritative, so drop it with a warning.
        if "DataType" in client_df.columns:
            logger.warning(
                "Client sidecar has a 'DataType' column; dropping it — the "
                "auto-detected type is authoritative."
            )
            client_df = client_df.drop(columns=["DataType"])
        merge_client_metadata(file_metadata, client_df)

    # Per detected type: canonicalize + date-infer + warn on its own row subset
    # and field groups. All (required + optional) groups drive canonicalization/
    # column selection; only the loud warning is restricted to required groups.
    field_groups_by_type: dict[str, list[list[str]]] = {}
    multi = len(detected_types) > 1
    for detected in detected_types:
        type_value = detected.value
        field_groups = _ALL_SIDECAR_FIELD_GROUPS.get(detected, [])
        field_groups_by_type[type_value] = field_groups
        required_field_groups = list(_REQUIRED_SIDECAR_FIELD_GROUPS.get(detected, []))
        labels_of_type = _labels_for(detected)
        subset = [
            (label, meta) for label, meta in file_metadata if label in labels_of_type
        ]
        _fill_missing_dates_from_filepath(subset, field_groups, display_to_key)
        _warn_missing_required_fields(
            subset, required_field_groups, type_label=type_value if multi else None
        )

    logger.info(f"Building sidecar DataFrame for {len(file_metadata)} file(s).")
    df = build_sidecar_df(
        file_metadata,
        field_groups_by_type=field_groups_by_type,
        types_by_label=types_by_label,
        full=full,
    )
    logger.info(
        f"Sidecar DataFrame: {len(df)} row(s) (including DEFAULT), "
        f"{len(df.columns)} column(s)."
    )

    if pc_metadata:
        # Record each point cloud's header CRS (with or without
        # --spatial-reference) so the linter can convert bounds into the goal CRS.
        _add_file_srs_column(df, pc_metadata)
        # Customer-facing flag: did each point cloud's effective CRS convert to
        # Web Mercator? (Uses the same fallback the linter does; rows marked "no"
        # are also reported as lint errors.)
        _add_crs_web_mercator_column(df, pc_metadata, spatial_reference)

    if spatial_reference:
        # Mixed content does BOTH: record the point-cloud fallback CRS on DEFAULT
        # and reproject image/video coordinates.
        if pc_metadata:
            logger.info(
                f"Recording fallback_srs={spatial_reference!r} in the DEFAULT row."
            )
            _add_spatial_reference_column(df, spatial_reference)
        iv_labels = _labels_for(*IMAGE_DATA_TYPES, *_VIDEO_DATA_TYPES)
        if iv_labels:
            logger.info(
                f"Reprojecting image/video coordinates from {spatial_reference!r} "
                "to EPSG:4326."
            )
            _reproject_dataframe(df, spatial_reference, only_labels=iv_labels)

    # Drop non-required columns no file populated (kept last so file_srs/
    # fallback_srs, added above, are considered). Required fields for the
    # detected types are protected even when blank — they stay as a checklist.
    required_canonicals = {
        group[0]
        for detected in detected_types
        for group in _REQUIRED_SIDECAR_FIELD_GROUPS.get(detected, [])
    }
    _drop_all_empty_columns(df, keep=required_canonicals)
    df = _move_blank_columns_to_end(df)

    return df, backend, detected_types, vector_keys


def _generate(
    directory: str,
    data_type: DataTypeEnum | None,
    output_filename: str,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool = False,
    local_copy: bool = False,
    spatial_reference: str | None = None,
) -> tuple[str, list[str], list[str]]:
    logger.info(
        f"Starting sidecar CSV generation — directory={directory!r} "
        f"data_type={data_type} output={output_filename!r} "
        f"client_sidecar={client_sidecar!r} client_schema={client_schema!r} "
        f"full={full} local_copy={local_copy} spatial_reference={spatial_reference!r}"
    )

    df, backend, _detected, vector_keys = _build_sidecar(
        directory=directory,
        data_type=data_type,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=full,
        spatial_reference=spatial_reference,
    )
    crs_failures = _crs_failures(df)

    local_copy_path: Path | None = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_csv = Path(tmp_dir) / output_filename
        df.to_csv(local_csv, index=False)
        logger.info(f"Writing sidecar CSV to {backend.display_root}/{output_filename}")
        sidecar_path = backend.write_output(output_filename, local_csv)
        if local_copy:
            local_copy_path = Path.cwd() / output_filename
            shutil.copy2(local_csv, local_copy_path)

    logger.info(f"Sidecar CSV written: {sidecar_path}")
    click.secho(
        f"\nSidecar written to: {sidecar_path}", fg="green", bold=True, err=True
    )
    if local_copy_path is not None:
        logger.info(f"Local copy written to: {local_copy_path}")
        click.secho(
            f"Local copy at: {local_copy_path}", fg="green", bold=True, err=True
        )

    return sidecar_path, vector_keys, crs_failures


def _format_replay_command(
    subcommand: list[str],
    directory: str,
    data_type: DataTypeEnum | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool,
    verbosity: "VerbosityChoice",
    spatial_reference: str | None,
    ignore_missing_orientation: bool,
    *,
    output_filename: str | None = None,
    local_copy: bool = False,
    coco: str | None = None,
) -> str:
    """Build the copy-pasteable replay command for ``sidecar`` or ``validate``.
    ``subcommand`` is the command words (e.g. ``["sidecar"]`` or
    ``["validate"]``); the save-only flags are emitted only when relevant
    (``validate`` passes neither). ``--datatype`` is emitted only when an
    explicit filter was chosen (auto-detect is the default).
    """
    parts = ["am-tools"]
    if verbosity == "verbose":
        parts.append("--verbose")
    elif verbosity == "silent":
        parts.append("--silent")
    parts += [*subcommand, "--directory", shlex.quote(directory)]
    if data_type is not None:
        parts += ["--datatype", data_type.value]
    if output_filename is not None:
        parts += ["--output-filename", shlex.quote(output_filename)]
    if client_sidecar:
        parts += ["--client-sidecar", shlex.quote(client_sidecar)]
    if client_schema is not None:
        parts += ["--client-schema", shlex.quote(client_schema)]
    if full:
        parts.append("--full")
    if local_copy:
        parts.append("--local-copy")
    if spatial_reference:
        parts += ["--spatial-reference", shlex.quote(spatial_reference)]
    if ignore_missing_orientation:
        parts.append("--ignore-missing-orientation")
    if coco:
        parts += ["--coco", shlex.quote(coco)]
    return " ".join(parts)


def _echo_replay_command(command: str) -> None:
    click.secho(
        "\nEquivalent command (copy & paste to skip the wizard next time):",
        fg="cyan",
        err=True,
    )
    click.secho(f"  {command}\n", fg="bright_cyan", bold=True, err=True)


class WizardResult(NamedTuple):
    """The values the interactive wizard resolves. ``output_filename`` and
    ``local_copy`` are only meaningful when the caller saves a sidecar; the
    ``validate`` command leaves them at their passed-in defaults and ignores them.
    """

    directory: str
    data_type: DataTypeEnum | None
    output_filename: str | None
    client_sidecar: str | None
    client_schema: str | None
    full: bool
    verbosity: "VerbosityChoice"
    local_copy: bool
    spatial_reference: str | None
    ignore_missing_orientation: bool
    coco: str | None


def _run_interactive_wizard(
    directory: str | None,
    data_type: DataTypeEnum | None,
    output_filename: str | None,
    client_sidecar: str | None,
    client_schema: str | None,
    full: bool,
    full_provided: bool,
    verbosity: "VerbosityChoice",
    verbosity_provided: bool,
    local_copy: bool,
    local_copy_provided: bool,
    spatial_reference: str | None,
    ignore_missing_orientation: bool,
    ignore_missing_orientation_provided: bool,
    coco: str | None = None,
    save: bool = True,
) -> WizardResult:
    """Prompt for any value the user didn't pass on the CLI.

    When ``save`` is False (the ``validate`` command), the output-filename and
    local-copy prompts are skipped because nothing is written to disk; those
    two values are returned unchanged.
    """
    questionary.print(
        "Interactive mode — press Ctrl+C at any time to cancel.\n",
        style="fg:ansibrightblack",
    )
    try:
        if directory is None:
            directory = _ask_directory()
        if data_type is None:
            data_type = _ask_data_type_filter()
        # Conditional prompts follow "show unless an explicit filter rules them
        # out": in auto mode (data_type is None) they all apply.
        if (
            data_type in (None, DataTypeEnum.oriented_image)
            and not ignore_missing_orientation_provided
        ):
            ignore_missing_orientation = _ask_ignore_missing_orientation()
        if save and output_filename is None:
            output_filename = _ask_output_filename()
        if client_sidecar is None:
            client_sidecar = _ask_client_sidecar()
        if client_schema is None and client_sidecar is not None:
            client_schema = _ask_client_schema()
        # COCO label impact only applies to imagery.
        if coco is None and (data_type is None or data_type in IMAGE_DATA_TYPES):
            coco = _ask_coco()
        # fallback_srs applies to point clouds; in auto mode --spatial-reference
        # also reprojects image/video coordinates, so offer it unless an explicit
        # non-point-cloud filter rules it out.
        if spatial_reference is None and (
            data_type is None or data_type in _POINT_CLOUD_DATA_TYPES
        ):
            spatial_reference = _ask_spatial_reference()
        if not full_provided:
            full = _ask_full()
        if save and not local_copy_provided and any(
            is_remote_uri(p) for p in (directory, client_sidecar, client_schema)
        ):
            local_copy = _ask_local_copy()
        if not verbosity_provided:
            verbosity = _ask_verbosity()
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    return WizardResult(
        directory=directory,
        data_type=data_type,
        output_filename=output_filename,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=full,
        verbosity=verbosity,
        local_copy=local_copy,
        spatial_reference=spatial_reference,
        ignore_missing_orientation=ignore_missing_orientation,
        coco=coco,
    )


def _run_generate(result: WizardResult, *, wizard_ran: bool) -> None:
    """Generate and lint a sidecar from a fully-resolved set of options.

    Shared by ``sidecar`` and the ``build-schema`` follow-on: it
    normalises the output filename, echoes the replay command (only when a
    wizard ran), writes the sidecar, then lints it. ``result.verbosity`` is
    assumed to have already been applied to the root logger by the caller.
    """
    directory = result.directory
    data_type = result.data_type

    output_filename = result.output_filename or "sidecar.csv"
    basename = Path(output_filename).name
    if basename != output_filename:
        logger.warning(
            f"--output-filename ignored directory portion of {output_filename!r}; "
            f"writing as {basename!r} in the input directory."
        )
        output_filename = basename

    if not directory:
        raise typer.BadParameter("Directory is required.")
    # data_type is an optional filter now — None means auto-detect.

    if wizard_ran:
        _echo_replay_command(
            _format_replay_command(
                ["sidecar"],
                directory=directory,
                data_type=data_type,
                output_filename=output_filename,
                client_sidecar=result.client_sidecar,
                client_schema=result.client_schema,
                full=result.full,
                verbosity=result.verbosity,
                local_copy=result.local_copy,
                spatial_reference=result.spatial_reference,
                ignore_missing_orientation=result.ignore_missing_orientation,
                coco=result.coco,
            )
        )

    try:
        sidecar_path, skipped_vector, crs_failures = _generate(
            directory=directory,
            data_type=data_type,
            output_filename=output_filename,
            client_sidecar=result.client_sidecar,
            client_schema=result.client_schema,
            full=result.full,
            local_copy=result.local_copy,
            spatial_reference=result.spatial_reference,
        )
    except Exception as e:
        _fail("Failed to generate sidecar CSV", e)

    logger.info("Linting generated sidecar…")
    try:
        report = lint_sidecar_file(
            sidecar_path,
            final=True,
            data_type=data_type,
            schema_path=None,
            input_files_path=directory,
            ignore_missing_orientation=result.ignore_missing_orientation,
            coco_path=result.coco,
        )
    except Exception as e:
        _fail("Failed to lint generated sidecar", e)

    typer.echo(report.render())
    _warn_skipped_vectors(skipped_vector)
    # A point cloud that can't reach Web Mercator is unusable to Flow; surface it
    # loudly at the very end and force a non-zero exit even if nothing else failed.
    _raise_crs_failures_loudly(crs_failures)
    if report.has_errors() or crs_failures:
        raise typer.Exit(code=1)


def offer_sidecar_after_schema(
    *,
    data_type: DataTypeEnum,
    client_sidecar: str,
    client_schema: str,
    verbosity: "VerbosityChoice" = "default",
    verbosity_provided: bool = False,
) -> None:
    """After ``build-schema`` writes a schema, offer to generate a sidecar now.

    The data type, the client sidecar CSV the schema was built from, and the
    schema file just written are carried straight into the sidecar wizard; only
    the remaining questions (scan directory, output filename, …) are asked.
    """
    from atomic_tools.cli import level_for

    proceed = questionary.confirm(
        "Build a sidecar now using this schema?",
        instruction=(
            "(Carries over the data type, the client sidecar, and the schema "
            "you just built; you'll be asked for the directory to scan and a "
            "few other options)"
        ),
        default=True,
    ).unsafe_ask()
    if not proceed:
        return

    result = _run_interactive_wizard(
        directory=None,
        data_type=data_type,
        output_filename=None,
        client_sidecar=client_sidecar,
        client_schema=client_schema,
        full=False,
        full_provided=False,
        verbosity=verbosity,
        verbosity_provided=verbosity_provided,
        local_copy=False,
        local_copy_provided=False,
        spatial_reference=None,
        ignore_missing_orientation=False,
        ignore_missing_orientation_provided=False,
        coco=None,
        save=True,
    )
    logging.getLogger().setLevel(
        level_for(
            verbose=result.verbosity == "verbose",
            silent=result.verbosity == "silent",
        )
    )
    _run_generate(result, wizard_ran=True)


def generate(
    ctx: typer.Context,
    directory: Annotated[
        str | None,
        typer.Option(
            help=(
                "Directory to scan. Either an object-store URI "
                "(s3://bucket/prefix, gs://..., az://...) or a local filesystem path."
            ),
        ),
    ] = None,
    data_type: Annotated[
        DataTypeFilter | None,
        typer.Option(
            "--datatype",
            "--data-type",
            help=(
                "Optional filter: restrict the scan to this data type. By default "
                "every file is auto-detected per file (recommended)."
            ),
            case_sensitive=False,
        ),
    ] = None,
    output_filename: Annotated[
        str | None,
        typer.Option(help="Filename for the generated sidecar CSV."),
    ] = None,
    client_sidecar: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional path to client-supplied sidecar data. May be an "
                "object-store URI (s3://bucket/key/file.csv) or a local path. "
                "Point it at a single CSV, or at a directory: when it's a "
                "directory, every CSV in a subdirectory BELOW it is merged into "
                "one (the directory itself is not scanned, since the generated "
                "sidecar is written there). All merged CSVs must share the same "
                "schema (column count); a mismatch aborts and names the bad "
                "file. Values are merged into the generated sidecar; client "
                "values win on conflict."
            ),
        ),
    ] = None,
    client_schema: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional client-supplied JSON schema describing how to "
                "normalise the client sidecar CSV (positional column names, "
                "per-client renames). Local path or object-store URI "
                "(s3://bucket/key/schema.json, gs://..., az://...). See "
                "schemas/column_names_example.json. If omitted, the client "
                "CSV is used as-is with no renames."
            ),
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help=(
                "Include every metadata field extracted from each file. By "
                "default, only the canonical/required fields for the data type "
                "are kept (plus blank columns for any missing required fields)."
            ),
        ),
    ] = False,
    local_copy: Annotated[
        bool,
        typer.Option(
            "--local-copy/--no-local-copy",
            help=(
                "Also write a copy of the sidecar to the current working "
                "directory. Useful when the input directory or client sidecar "
                "is in object storage and you want a local copy to inspect. "
                "When unset, the wizard prompts only if a remote (s3://, "
                "gs://, az://) path was given."
            ),
        ),
    ] = False,
    spatial_reference: Annotated[
        str | None,
        typer.Option(
            "--spatial-reference",
            "--spatial_reference",
            help=(
                "CRS of the source coordinates (e.g. 'EPSG:32612' or '32612'). "
                "For images, lat/lon (and altitude) are treated as X/Y/Z "
                "in this CRS and reprojected to EPSG:4326. For point clouds, a "
                "'fallback_srs' column is added with this value in the "
                "DEFAULT row."
            ),
        ),
    ] = None,
    ignore_missing_orientation: Annotated[
        bool,
        typer.Option(
            "--ignore-missing-orientation",
            help=(
                "Only meaningful for --datatype oriented_image. By default, "
                "missing orientation (Pitch/Heading/Roll) in the generated "
                "sidecar is an error; pass this to downgrade it to a warning "
                "(the images still process, appearing in Lens without orientation)."
            ),
        ),
    ] = False,
    coco: Annotated[
        str | None,
        typer.Option(
            "--coco",
            help=(
                "Optional COCO label file (local path, s3://… URI, or a directory "
                "containing one). Image data types only. After linting, reports "
                "how many labels sit on images with missing/zero-size metadata "
                "(degraded/unusable/not_on_disk), and adds those tiers to the "
                "failed-rows CSV (--report)."
            ),
        ),
    ] = None,
) -> None:
    """Scan a directory, extract per-file metadata, and write a sidecar CSV.

    With no flags, prompts interactively for each value. Any flag that is
    passed on the command line is used as-is and not prompted for.
    """
    from atomic_tools.cli import Verbosity, level_for

    verbosity_state: Verbosity = ctx.ensure_object(Verbosity)
    verbosity_choice = verbosity_state.choice
    verbosity_provided = verbosity_state.verbose or verbosity_state.silent

    # --datatype is an optional filter; convert it to the full enum immediately.
    data_type_enum: DataTypeEnum | None = (
        DataTypeEnum(data_type.value) if data_type is not None else None
    )

    # The wizard now only runs when the directory is missing; --datatype being
    # unset just means auto-detect, not "ask me".
    wizard_ran = directory is None
    if wizard_ran:
        full_provided = (
            ctx.get_parameter_source("full") == click.core.ParameterSource.COMMANDLINE
        )
        local_copy_provided = (
            ctx.get_parameter_source("local_copy")
            == click.core.ParameterSource.COMMANDLINE
        )
        ignore_orientation_provided = (
            ctx.get_parameter_source("ignore_missing_orientation")
            == click.core.ParameterSource.COMMANDLINE
        )
        result = _run_interactive_wizard(
            directory,
            data_type_enum,
            output_filename,
            client_sidecar,
            client_schema,
            full,
            full_provided,
            verbosity_choice,
            verbosity_provided,
            local_copy,
            local_copy_provided,
            spatial_reference,
            ignore_missing_orientation,
            ignore_orientation_provided,
            coco=coco,
        )
        verbosity_choice = result.verbosity
        logging.getLogger().setLevel(
            level_for(
                verbose=verbosity_choice == "verbose",
                silent=verbosity_choice == "silent",
            )
        )
    else:
        result = WizardResult(
            directory=directory,
            data_type=data_type_enum,
            output_filename=output_filename,
            client_sidecar=client_sidecar,
            client_schema=client_schema,
            full=full,
            verbosity=verbosity_choice,
            local_copy=local_copy,
            spatial_reference=spatial_reference,
            ignore_missing_orientation=ignore_missing_orientation,
            coco=coco,
        )

    _run_generate(result, wizard_ran=wizard_ran)
