"""Plugin loading and management."""

from pathlib import Path


def resolve_plugin_paths(plugin_paths: list[str]) -> list[Path]:
    """Resolve plugin paths to absolute paths.

    Args:
        plugin_paths: List of local plugin paths

    Returns:
        List of resolved Path objects

    Raises:
        FileNotFoundError: If a plugin path doesn't exist
        ValueError: If a plugin path is not a directory
    """
    resolved = []
    for path_str in plugin_paths:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Plugin not found: {path_str}")
        if not path.is_dir():
            raise ValueError(f"Plugin path is not a directory: {path_str}")
        resolved.append(path)
    return resolved
