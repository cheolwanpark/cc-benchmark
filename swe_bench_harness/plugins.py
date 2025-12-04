"""Plugin loading and management.

This module handles loading plugins from local paths and GitHub URLs,
with automatic cleanup of cloned repositories.
"""

import atexit
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class PluginLoader:
    """Manages plugin loading from local paths and GitHub URLs.

    Uses atexit handler as safety net for cleanup on crashes.
    """

    def __init__(self) -> None:
        """Initialize the plugin loader."""
        self._temp_dirs: list[Path] = []
        self._cleanup_registered = False
        # Register cleanup on process exit (safety net for crashes)
        atexit.register(self._atexit_cleanup)
        self._cleanup_registered = True

    def load(self, uri: str) -> str:
        """Load plugin and return local path.

        Args:
            uri: Local path or GitHub URL

        Returns:
            Resolved local path to plugin

        Raises:
            FileNotFoundError: If local path doesn't exist
            ValueError: If local path is not a directory
            subprocess.CalledProcessError: If git clone fails
            subprocess.TimeoutExpired: If git clone times out
        """
        if uri.startswith("https://github.com/"):
            return self._clone_github(uri)

        # Local path - resolve and validate
        path = Path(uri).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Plugin not found: {uri}")
        if not path.is_dir():
            raise ValueError(f"Plugin path is not a directory: {uri}")
        return str(path)

    def _clone_github(self, url: str) -> str:
        """Clone GitHub repo to temp directory (shallow clone).

        Args:
            url: GitHub repository URL

        Returns:
            Path to cloned repository
        """
        # Create parent temp directory, let git create the clone directory
        parent_dir = Path(tempfile.mkdtemp(prefix="swe-plugin-parent-"))
        clone_dir = parent_dir / "plugin"
        self._temp_dirs.append(parent_dir)

        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(clone_dir)],
            check=True,
            capture_output=True,
            timeout=120,  # 2 minute timeout for clone
        )
        return str(clone_dir)

    def _atexit_cleanup(self) -> None:
        """Atexit handler - just cleans up, doesn't unregister."""
        self._do_cleanup()

    def cleanup(self) -> None:
        """Remove all cloned temp directories and unregister atexit handler."""
        self._do_cleanup()

        # Unregister atexit handler after explicit cleanup
        if self._cleanup_registered:
            try:
                atexit.unregister(self._atexit_cleanup)
                self._cleanup_registered = False
            except (ValueError, TypeError):
                pass  # Handler already unregistered or not found

    def _do_cleanup(self) -> None:
        """Internal cleanup implementation."""
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except (OSError, PermissionError):
                pass  # Best effort cleanup
        self._temp_dirs.clear()


@contextmanager
def plugin_context() -> Generator[PluginLoader, None, None]:
    """Context manager for plugin lifecycle.

    Ensures cleanup even on exceptions/crashes via:
    1. Context manager finally block
    2. atexit handler as safety net

    Yields:
        PluginLoader instance for loading plugins
    """
    loader = PluginLoader()
    try:
        yield loader
    finally:
        loader.cleanup()
