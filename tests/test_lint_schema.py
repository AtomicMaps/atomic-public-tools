"""Tests for `am-tools lint schema`."""

from pathlib import Path

from atomic_tools.validators.schema import lint_schema_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path, content: str) -> Path:
    p = tmp_path / "schema.json"
    p.write_text(content, encoding="utf-8")
    return p


def test_valid_example_schema_passes():
    report = lint_schema_file(REPO_ROOT / "schemas" / "column_names_example.json")
    assert not report.has_errors(), report.render()


def test_missing_file(tmp_path):
    report = lint_schema_file(tmp_path / "no_such.json")
    assert report.has_errors()
    assert any("does not exist" in f.message for f in report.errors())


def test_invalid_json(tmp_path):
    p = _write(tmp_path, "{")
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("Invalid JSON" in f.message for f in report.errors())


def test_root_not_object(tmp_path):
    p = _write(tmp_path, "[]")
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("must be a JSON object" in f.message for f in report.errors())


def test_missing_both_keys(tmp_path):
    p = _write(tmp_path, "{}")
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("at least one" in f.message for f in report.errors())


def test_only_column_names_passes(tmp_path):
    p = _write(tmp_path, '{"column_names": ["A", "B"]}')
    report = lint_schema_file(p)
    assert not report.has_errors(), report.render()


def test_only_column_name_mapping_passes(tmp_path):
    p = _write(tmp_path, '{"column_name_mapping": {"a": "GPSLatitude"}}')
    report = lint_schema_file(p)
    assert not report.has_errors(), report.render()


def test_comments_not_supported(tmp_path):
    p = _write(tmp_path, '{\n  // a comment\n  "column_names": []\n}')
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("Invalid JSON" in f.message for f in report.errors())


def test_column_names_not_a_list(tmp_path):
    p = _write(tmp_path, '{"column_names": "not a list"}')
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("must be a list" in f.message for f in report.errors())


def test_column_names_non_string_entries(tmp_path):
    p = _write(tmp_path, '{"column_names": ["A", 1, "B", null]}')
    report = lint_schema_file(p)
    assert report.has_errors()
    msgs = [f.message for f in report.errors()]
    assert any("'column_names[1]'" in m for m in msgs)
    assert any("'column_names[3]'" in m for m in msgs)


def test_column_names_duplicates_warn(tmp_path):
    p = _write(tmp_path, '{"column_names": ["A", "B", "A"]}')
    report = lint_schema_file(p)
    assert not report.has_errors()
    assert any("Duplicate" in f.message for f in report.warnings())


def test_column_name_mapping_not_object(tmp_path):
    p = _write(tmp_path, '{"column_name_mapping": ["a", "b"]}')
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("must be an object" in f.message for f in report.errors())


def test_rename_target_collision(tmp_path):
    p = _write(
        tmp_path,
        '{"column_name_mapping": {"a": "GPSLatitude", "b": "GPSLatitude"}}',
    )
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("rename to the same target" in f.message for f in report.errors())


def test_rename_identity_warns(tmp_path):
    p = _write(tmp_path, '{"column_name_mapping": {"X": "X"}}')
    report = lint_schema_file(p)
    assert any("Identity rename" in f.message for f in report.warnings())


def test_unknown_top_level_key_errors(tmp_path):
    p = _write(tmp_path, '{"columnNames": ["A"]}')
    report = lint_schema_file(p)
    assert report.has_errors()
    assert any("Unknown top-level key" in f.message for f in report.errors())


def test_cross_check_warns_on_unknown_rename_source(tmp_path):
    p = _write(
        tmp_path,
        '{"column_names": ["A"], "column_name_mapping": {"B": "GPSLatitude"}}',
    )
    report = lint_schema_file(p)
    assert not report.has_errors()
    assert any("not in 'column_names'" in f.message for f in report.warnings())
