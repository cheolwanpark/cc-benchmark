"""Prompt template management using Jinja2.

This module provides structured, customizable prompts for the benchmark agent.
Templates can be loaded from files or used inline with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment

# Template directory structure
TEMPLATE_DIR = Path(__file__).parent / "templates"


class PromptManager:
    """Manages prompt templates with Jinja2's sandboxed environment."""

    def __init__(self, template_dir: Path | None = None):
        """Initialize with optional custom template directory.

        Args:
            template_dir: Optional path to custom template directory.
                         Defaults to cc_benchmark/templates/

        Raises:
            FileNotFoundError: If template directory doesn't exist
        """
        self.template_dir = template_dir or TEMPLATE_DIR

        if not self.template_dir.exists():
            raise FileNotFoundError(
                f"Template directory not found: {self.template_dir}. "
                "Ensure templates are included in the package."
            )

        if not self.template_dir.is_dir():
            raise NotADirectoryError(f"Template path is not a directory: {self.template_dir}")

        # Use file-based templates with sandboxed environment
        # Note: autoescape disabled for plain-text LLM prompts (not HTML)
        self.env = SandboxedEnvironment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=False,  # Plain text prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_system_prompt(self, **kwargs: Any) -> str:
        """Render system prompt from template file.

        Args:
            **kwargs: Additional template variables (currently unused for system prompt)

        Returns:
            Rendered system prompt string

        Raises:
            TemplateNotFound: If system_prompt.j2 doesn't exist
        """
        template = self.env.get_template("system_prompt.j2")
        return template.render(**kwargs)

    def render_user_message(self, **kwargs: Any) -> str:
        """Render user message from template file.

        Args:
            **kwargs: Template variables (repo, problem, fail_to_pass, base_commit, hints_text)

        Returns:
            Rendered user message string

        Raises:
            TemplateNotFound: If user_message.j2 doesn't exist
        """
        template = self.env.get_template("user_message.j2")
        return template.render(**kwargs)


# Singleton manager instance for efficiency (avoid recreating Jinja2 env every call)
_default_manager: PromptManager | None = None


def _get_default_manager() -> PromptManager:
    """Get or create the default PromptManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager


def build_system_prompt(**kwargs: Any) -> str:
    """Build system prompt with default manager.

    Args:
        **kwargs: Additional template variables (currently unused)

    Returns:
        Rendered system prompt string
    """
    return _get_default_manager().render_system_prompt(**kwargs)


def build_user_message(
    repo: str,
    problem: str,
    fail_to_pass: str,
    base_commit: str = "",
    hints_text: str = "",
    **kwargs: Any,
) -> str:
    """Build user message with default manager.

    Args:
        repo: Repository name (e.g., "owner/repo")
        problem: Problem statement describing the issue
        fail_to_pass: Tests that should pass after the fix
        base_commit: Git commit hash (optional)
        hints_text: Optional hints for solving the problem
        **kwargs: Additional template variables

    Returns:
        Rendered user message string
    """
    return _get_default_manager().render_user_message(
        repo=repo,
        problem=problem,
        fail_to_pass=fail_to_pass,
        base_commit=base_commit,
        hints_text=hints_text,
        **kwargs,
    )
