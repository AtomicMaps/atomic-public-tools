"""AWS S3 console/HTTPS URLs are translated to s3:// URIs (with a warning)."""

from __future__ import annotations

import logging

import pytest

from atomic_tools.io.storage import console_url_to_s3_uri, normalize_directory


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Web console — bucket in the path, prefix in the query string.
        (
            "https://us-west-2.console.aws.amazon.com/s3/buckets/atomic-lens-ephemeral"
            "?region=us-west-2&prefix=shared_data/core-processing-data/"
            "&showversions=false",
            "s3://atomic-lens-ephemeral/shared_data/core-processing-data",
        ),
        # Console with no prefix -> bucket root.
        (
            "https://s3.console.aws.amazon.com/s3/buckets/my-bucket",
            "s3://my-bucket",
        ),
        # Virtual-hosted-style HTTPS URL.
        (
            "https://my-bucket.s3.us-west-2.amazonaws.com/a/b/",
            "s3://my-bucket/a/b",
        ),
        # Path-style HTTPS URL.
        (
            "https://s3.us-west-2.amazonaws.com/my-bucket/a/b",
            "s3://my-bucket/a/b",
        ),
    ],
)
def test_console_url_to_s3_uri(url, expected):
    assert console_url_to_s3_uri(url) == expected


@pytest.mark.parametrize(
    "value",
    [
        "s3://already/an-uri",
        "/Users/me/local/dir",
        "relative/path",
        "https://example.com/not-s3",
    ],
)
def test_non_console_urls_pass_through(value):
    assert console_url_to_s3_uri(value) is None
    assert normalize_directory(value) == value


def test_normalize_directory_warns_on_translation(caplog):
    url = (
        "https://us-west-2.console.aws.amazon.com/s3/buckets/bkt"
        "?prefix=some/prefix/"
    )
    with caplog.at_level(logging.WARNING):
        result = normalize_directory(url)
    assert result == "s3://bkt/some/prefix"
    assert "s3://bkt/some/prefix" in caplog.text
