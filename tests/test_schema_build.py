"""Tests for `am-tools schema build` (pure helpers + CLI registration)."""

import json

from typer.testing import CliRunner

from atomic_tools.cli import app
from atomic_tools.commands import schema as schema_cmd
from atomic_tools.commands.schema import (
    _default_schema_filename,
    build_schema,
    canonical_label,
    canonical_names_for,
    column_stats,
    exact_or_alias_match,
    field_tier,
    rank_canonicals,
)
from atomic_tools.utils.utils import DataTypeEnum

runner = CliRunner()


class _Answer:
    """Stand-in for a questionary prompt whose `.unsafe_ask()` is pre-baked."""

    def __init__(self, value):
        self._value = value

    def unsafe_ask(self):
        return self._value


def _patch_prompts(monkeypatch, answers):
    """Make questionary text/select/confirm pop the next canned answer."""
    queue = list(answers)

    def _next(*_args, **_kwargs):
        return _Answer(queue.pop(0))

    for name in ("text", "select", "confirm"):
        monkeypatch.setattr(schema_cmd.questionary, name, _next)


# ---- canonical names -------------------------------------------------------


def test_canonical_names_filename_first_and_include_gps():
    names = canonical_names_for(DataTypeEnum.oriented_image)
    assert names[0] == "Filename"
    for expected in ("GPSLatitude", "GPSLongitude", "GPSAltitude", "CreateDate"):
        assert expected in names


def test_canonical_names_point_cloud():
    names = canonical_names_for(DataTypeEnum.point_cloud)
    assert "bounds.minx" in names
    assert "num_points" in names


# ---- referenced (non-required) fields --------------------------------------


def test_curated_referenced_fields_offered_by_default():
    names = canonical_names_for(DataTypeEnum.oriented_image)
    for expected in ("FocalLength", "Make", "Model", "GPSXYAccuracy"):
        assert expected in names


def test_curated_referenced_label_lists_aliases():
    label = canonical_label("FocalLength", DataTypeEnum.oriented_image)
    assert label.startswith("FocalLength (")
    assert "FocalLengthIn35mmFormat" in label


def test_comprehensive_fields_only_with_full():
    default_names = canonical_names_for(DataTypeEnum.oriented_image)
    full_names = canonical_names_for(DataTypeEnum.oriented_image, full=True)
    assert "CalibratedOpticalCenterX" not in default_names
    assert "CalibratedOpticalCenterX" in full_names
    # Curated fields remain available under --full.
    assert "FocalLength" in full_names


def test_referenced_alias_resolves_to_canonical():
    assert (
        exact_or_alias_match("FocalLengthIn35mmFormat", DataTypeEnum.oriented_image)
        == "FocalLength"
    )


def test_comprehensive_alias_needs_full():
    assert exact_or_alias_match("UserComment", DataTypeEnum.spherical_image) is None
    assert (
        exact_or_alias_match("UserComment", DataTypeEnum.spherical_image, full=True)
        == "CameraOrientation"
    )


def test_extended_date_aliases_resolve_to_date_canonical():
    for alias in ("FirstPhotoDate", "LastPhotoDate", "GPSTimeStamp"):
        assert (
            exact_or_alias_match(alias, DataTypeEnum.oriented_image) == "CreateDate"
        )


# ---- fuzzy ranking + matching ---------------------------------------------


def test_rank_puts_fuzzy_match_first():
    ranked = rank_canonicals("Latitude [Degrees]", DataTypeEnum.oriented_image)
    assert ranked[0] == "GPSLatitude"


def test_rank_is_stable_for_unmatched_header():
    ranked = rank_canonicals("qqzzxx", DataTypeEnum.oriented_image)
    assert ranked == canonical_names_for(DataTypeEnum.oriented_image)


def test_exact_match_on_canonical():
    assert exact_or_alias_match("GPSLatitude", DataTypeEnum.oriented_image) == "GPSLatitude"


def test_match_on_alias_resolves_to_canonical():
    # `Yaw` is an alias of `Heading` in the oriented_image optional groups.
    assert exact_or_alias_match("Yaw", DataTypeEnum.oriented_image) == "Heading"


def test_no_match_returns_none():
    assert exact_or_alias_match("nonsense", DataTypeEnum.oriented_image) is None


# ---- field tiers (color + ranking) ----------------------------------------


def test_field_tier_classifies_each_kind():
    dt = DataTypeEnum.oriented_image
    assert field_tier("Filename", dt) == "required"  # implicit match column
    assert field_tier("GPSLatitude", dt) == "required"
    assert field_tier("CreateDate", dt) == "required"
    assert field_tier("Heading", dt) == "optional"
    assert field_tier("FocalLength", dt) == "referenced"  # curated referenced
    assert field_tier("CalibratedOpticalCenterX", dt) == "full"  # --full only


def test_rank_orders_by_tier_when_unmatched():
    ranked = rank_canonicals("qqzzxx", DataTypeEnum.oriented_image, full=True)
    tiers = [field_tier(c, DataTypeEnum.oriented_image) for c in ranked]
    weight = {"required": 0, "optional": 1, "referenced": 2, "full": 3}
    weights = [weight[t] for t in tiers]
    assert weights == sorted(weights), tiers


def test_rank_shows_guess_then_required_then_optional():
    # `Make` is a referenced field but a confident guess for this header, so it
    # leads; the incidental "cam" overlaps against Camera* fields must NOT jump
    # ahead of the required fields.
    dt = DataTypeEnum.oriented_image
    ranked = rank_canonicals("Cam Make", dt)
    assert ranked[0] == "Make"  # the guess leads
    rest = ranked[1:]
    # Everything after the guess is grouped required -> optional -> referenced.
    tiers = [field_tier(c, dt) for c in rest]
    weight = {"required": 0, "optional": 1, "referenced": 2, "full": 3}
    weights = [weight[t] for t in tiers]
    assert weights == sorted(weights), rest
    # Required fields come before the incidental Camera* optional matches.
    assert rest.index("GPSLatitude") < rest.index("Heading")


# ---- dropdown labels -------------------------------------------------------


def test_canonical_label_lists_aliases_in_parens():
    label = canonical_label("Heading", DataTypeEnum.oriented_image)
    assert label.startswith("Heading (")
    assert "Yaw" in label
    assert label.endswith(")")


def test_canonical_label_no_aliases_is_bare_name():
    assert canonical_label("GPSLatitude", DataTypeEnum.oriented_image) == "GPSLatitude"
    assert canonical_label("Filename", DataTypeEnum.oriented_image) == "Filename"


# ---- column stats ----------------------------------------------------------


def test_column_stats_counts_blanks_and_unique():
    stats = column_stats(["a", "b", "", "a", "nan", "  ", "c"])
    assert stats["total"] == 7
    assert stats["blank"] == 3  # "", "nan", "  "
    assert stats["non_blank"] == 4
    assert stats["unique"] == 3  # a, b, c
    assert stats["samples"] == ["a", "b", "c"]


# ---- default filename ------------------------------------------------------


def test_default_schema_filename_from_local_csv():
    assert _default_schema_filename("/data/clientA.csv") == "clientA_schema.json"


def test_default_schema_filename_strips_csv_case_insensitively():
    assert _default_schema_filename("Sidecar.CSV") == "Sidecar_schema.json"


def test_default_schema_filename_from_remote_uri():
    assert (
        _default_schema_filename("s3://bucket/prefix/run-01.csv")
        == "run-01_schema.json"
    )


def test_default_schema_filename_non_csv_extension():
    assert _default_schema_filename("/data/export.tsv") == "export_schema.json"


# ---- schema building -------------------------------------------------------


def test_build_headered_skips_identity_renames():
    decisions = [
        ("Filename", "Filename"),  # left as-is -> omitted
        ("Lat", "GPSLatitude"),
        ("Lon", "GPSLongitude"),
    ]
    schema = build_schema(decisions, has_header=True)
    assert schema == {
        "column_name_mapping": {"Lat": "GPSLatitude", "Lon": "GPSLongitude"}
    }


def test_build_headerless_is_positional_list():
    decisions = [
        ("column_1", "Filename"),
        ("column_2", "GPSLatitude"),
        ("column_3", "column_3"),  # placeholder kept
    ]
    schema = build_schema(decisions, has_header=False)
    assert schema == {"column_names": ["Filename", "GPSLatitude", "column_3"]}


# ---- CLI registration ------------------------------------------------------


def test_schema_group_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "schema" in result.stdout


def test_schema_build_help():
    result = runner.invoke(app, ["schema", "build", "--help"])
    assert result.exit_code == 0
    assert "CSV" in result.stdout


# ---- end-to-end wizard (mocked prompts) ------------------------------------


def test_build_headered_end_to_end(tmp_path, monkeypatch):
    csv = tmp_path / "client.csv"
    csv.write_text(
        "Filename,Latitude [Degrees],Longitude [Degrees]\n"
        "a.jpg,40.1,-74.2\n"
        "b.jpg,,\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    _patch_prompts(
        monkeypatch,
        answers=[
            DataTypeEnum.oriented_image.value,  # data type
            True,  # has header?
            "Filename",  # col Filename
            "GPSLatitude",  # col Latitude [Degrees]
            "GPSLongitude",  # col Longitude [Degrees]
            schema_cmd._SAVE_LOCAL,  # save target
            "schema.json",  # filename
            str(out),  # local path
        ],
    )
    result = runner.invoke(app, ["schema", "build", str(csv)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written == {
        "column_name_mapping": {
            "Latitude [Degrees]": "GPSLatitude",
            "Longitude [Degrees]": "GPSLongitude",
        }
    }


def test_build_full_offers_comprehensive_field(tmp_path, monkeypatch):
    csv = tmp_path / "client.csv"
    csv.write_text(
        "Filename,cx\na.jpg,1024\nb.jpg,1024\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    _patch_prompts(
        monkeypatch,
        answers=[
            DataTypeEnum.oriented_image.value,  # data type
            True,  # has header?
            "Filename",  # col Filename
            "CalibratedOpticalCenterX",  # col cx -> comprehensive field
            schema_cmd._SAVE_LOCAL,  # save target
            "schema.json",  # filename
            str(out),  # local path
        ],
    )
    result = runner.invoke(app, ["schema", "build", "--full", str(csv)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written == {"column_name_mapping": {"cx": "CalibratedOpticalCenterX"}}


def test_build_headerless_end_to_end(tmp_path, monkeypatch):
    csv = tmp_path / "headerless.csv"
    csv.write_text("a.jpg,40.1,-74.2\nb.jpg,41.0,-75.0\n", encoding="utf-8")
    out = tmp_path / "out.json"
    _patch_prompts(
        monkeypatch,
        answers=[
            DataTypeEnum.oriented_image.value,  # data type
            False,  # has header?
            "Filename",  # column_1
            "GPSLatitude",  # column_2
            "GPSLongitude",  # column_3
            schema_cmd._SAVE_LOCAL,  # save target
            "schema.json",  # filename
            str(out),  # local path
        ],
    )
    result = runner.invoke(app, ["schema", "build", str(csv)])
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written == {"column_names": ["Filename", "GPSLatitude", "GPSLongitude"]}
