"""Actionable AWS auth error types and help-text rendering.

When `am-tools` talks to S3 we want the end-user (a client running the CLI)
to get a short, copy-pasteable next step rather than a raw obstore/boto3
traceback. The exceptions here are caught at the CLI top-level and rendered
in bright red after the stack trace.
"""

from __future__ import annotations

import os
import shutil

import click

_SSO_GUIDE_URL = (
    "https://dev.to/slsbytheodo/understand-the-aws-sso-login-configuration-4am7"
)
_AWS_CLI_INSTALL_URL = (
    "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
)

_OPERATION_TO_IAM_ACTION = {
    "list": "s3:ListBucket",
    "download": "s3:GetObject",
    "upload": "s3:PutObject",
}


def aws_cli_installed() -> bool:
    """Return True if an ``aws`` executable is on PATH."""
    return shutil.which("aws") is not None


def current_profile() -> str | None:
    """Return the active AWS profile name, or None if the default profile is in use.

    Resolution order matches boto3: explicit env var first, then fall back to
    ``None`` (which boto3 treats as the ``default`` profile).
    """
    return os.environ.get("AWS_PROFILE") or os.environ.get("AWS_DEFAULT_PROFILE")


def required_iam_action_for(operation: str) -> str:
    return _OPERATION_TO_IAM_ACTION.get(operation, "s3:*")


class S3AuthError(RuntimeError):
    """Base class for actionable AWS auth failures.

    Subclasses implement ``help_text()`` — a multi-line block the CLI prints
    in bright red after the stack trace, so the user sees a clear next step.
    """

    def help_text(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class S3CredentialsMissingError(S3AuthError):
    """Raised when ``boto3.Session.get_credentials()`` returns ``None``."""

    def __init__(self, profile: str | None) -> None:
        self.profile = profile
        super().__init__(
            "No AWS credentials found for "
            f"profile={profile or 'default'!s}. Run `aws sso login` and retry."
        )

    def help_text(self) -> str:
        profile = self.profile or "default"
        lines: list[str] = []
        if not aws_cli_installed():
            lines.append(
                "The AWS CLI is not installed on this machine. Install it "
                "before running `aws sso login`:"
            )
            lines.append(f"  {_AWS_CLI_INSTALL_URL}")
            lines.append("")
        lines.append(
            "AWS credentials not found. boto3 could not locate any usable "
            f"credentials for this session (profile: {profile})."
        )
        lines.append("")
        lines.append("To authenticate with AWS SSO:")
        lines.append(f"  1. aws sso login --profile {profile}")
        lines.append("  2. Re-run this command.")
        lines.append("")
        lines.append(
            "See the one-time AWS SSO setup walkthrough at:"
        )
        lines.append(f"  {_SSO_GUIDE_URL}")
        return "\n".join(lines)


class S3PermissionDeniedError(S3AuthError):
    """Raised when the active identity is authenticated but denied by S3."""

    def __init__(
        self,
        *,
        profile: str | None,
        bucket: str,
        prefix: str,
        operation: str,
    ) -> None:
        self.profile = profile
        self.bucket = bucket
        self.prefix = prefix
        self.operation = operation
        super().__init__(
            f"S3 {operation} denied for profile={profile or 'default'!s} "
            f"on s3://{bucket}/{prefix}"
        )

    def _resource_arn(self) -> str:
        if self.operation == "list":
            return f"arn:aws:s3:::{self.bucket}"
        suffix = f"{self.prefix}/*" if self.prefix else "*"
        return f"arn:aws:s3:::{self.bucket}/{suffix}"

    def help_text(self) -> str:
        profile = self.profile or "default"
        action = required_iam_action_for(self.operation)
        prefix_display = f"/{self.prefix}" if self.prefix else ""
        lines = [
            "S3 request denied. The active AWS identity (profile: "
            f"{profile}) is authenticated but lacks permission for this "
            "operation.",
            "",
            f"  Bucket:    s3://{self.bucket}{prefix_display}",
            f"  Operation: {self.operation}",
            f"  Likely missing IAM action: {action}",
            f"    on resource {self._resource_arn()}",
            "",
            "Ask your AWS administrator to grant this action, or switch to "
            "a profile that already has it (AWS_PROFILE=<other> am-tools …).",
        ]
        return "\n".join(lines)


def print_help_block(error: S3AuthError) -> None:
    """Print an S3AuthError's help text in bright red to stderr.

    Matches the style used elsewhere in the CLI (see commands/sidecar.py
    for the missing-required-fields warning).
    """
    click.secho(error.help_text(), fg="bright_red", bold=True, err=True, color=True)


def find_auth_error(exc: BaseException | None) -> S3AuthError | None:
    """Walk the ``__cause__``/``__context__`` chain looking for an S3AuthError."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, S3AuthError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None
