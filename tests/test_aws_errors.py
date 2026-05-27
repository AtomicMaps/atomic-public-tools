from __future__ import annotations

import pytest

from atomic_tools.utils import aws_errors
from atomic_tools.utils.aws_errors import (
    S3AuthError,
    S3CredentialsMissingError,
    S3PermissionDeniedError,
    find_auth_error,
    required_iam_action_for,
)


def test_credentials_missing_help_links_to_sso_guide(monkeypatch):
    monkeypatch.setattr(aws_errors, "aws_cli_installed", lambda: True)
    err = S3CredentialsMissingError(profile="my-profile")
    text = err.help_text()
    assert "aws sso login --profile my-profile" in text
    assert "dev.to/slsbytheodo" in text
    # AWS CLI install link should NOT be present when CLI is installed
    assert "getting-started-install" not in text


def test_credentials_missing_help_prepends_cli_install_link_when_missing(monkeypatch):
    monkeypatch.setattr(aws_errors, "aws_cli_installed", lambda: False)
    err = S3CredentialsMissingError(profile=None)
    text = err.help_text()
    # CLI install instructions come first
    assert text.startswith("The AWS CLI is not installed")
    assert "getting-started-install" in text
    # Falls back to "default" profile when none provided
    assert "aws sso login --profile default" in text
    assert "dev.to/slsbytheodo" in text


def test_permission_denied_help_includes_profile_bucket_and_action():
    err = S3PermissionDeniedError(
        profile="data-rw",
        bucket="my-bucket",
        prefix="acme/2024",
        operation="list",
    )
    text = err.help_text()
    assert "profile: data-rw" in text
    assert "s3://my-bucket/acme/2024" in text
    assert "s3:ListBucket" in text
    assert "arn:aws:s3:::my-bucket" in text


@pytest.mark.parametrize(
    ("operation", "expected_action"),
    [
        ("list", "s3:ListBucket"),
        ("download", "s3:GetObject"),
        ("upload", "s3:PutObject"),
        ("unknown", "s3:*"),
    ],
)
def test_required_iam_action_for(operation, expected_action):
    assert required_iam_action_for(operation) == expected_action


def test_permission_denied_resource_arn_for_object_operations():
    err = S3PermissionDeniedError(
        profile=None,
        bucket="b",
        prefix="x/y",
        operation="download",
    )
    text = err.help_text()
    assert "arn:aws:s3:::b/x/y/*" in text


def test_find_auth_error_walks_cause_chain():
    auth = S3PermissionDeniedError(
        profile="p", bucket="b", prefix="", operation="list"
    )
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = auth
    assert find_auth_error(wrapped) is auth


def test_find_auth_error_returns_none_for_non_auth_chain():
    err = RuntimeError("nothing to see")
    assert find_auth_error(err) is None


def test_s3_auth_error_is_runtime_error():
    # Subclassing RuntimeError matters because validators/sidecar.py catches it
    assert issubclass(S3AuthError, RuntimeError)
