import os
from pathlib import Path


class WorkspaceGuard:
    """
    Restricts all file system actions within a configured root directory
    and rejects any directory traversal attempts (`../` or escaping path).
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def validate_path(self, path: str) -> Path:
        """
        Validates and resolves a file path.
        Raises PermissionError if the resolved path is outside workspace_root.
        """
        # Explicit check for traversal string components
        clean_path_str = str(path)
        if ".." in clean_path_str.split("/") or ".." in clean_path_str.split("\\"):
            raise PermissionError(f"Directory traversal detected in path: '{path}'")

        target = (self.workspace_root / path).resolve()

        # Check boundary containment
        try:
            target.relative_to(self.workspace_root)
        except ValueError:
            raise PermissionError(
                f"Path '{path}' resolves to '{target}' which is outside workspace root '{self.workspace_root}'"
            )

        return target

    def is_safe_path(self, path: str) -> bool:
        try:
            self.validate_path(path)
            return True
        except PermissionError:
            return False
