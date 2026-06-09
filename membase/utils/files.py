from pathlib import Path
import importlib
import sys
from typing import Callable, Literal


def import_function_from_path(function_path: str) -> Callable:
    """Dynamically import a function from a module path.
    
    This function supports two formats:
    1. Standard Python module path: "module.submodule.function_name"
    2. File path: "path/to/file.py:function_name"
    
    Args:
        function_path (`str`):
            The function path in the format "module.submodule.function_name"
            or "path/to/file.py:function_name" for local files.
    
    Returns:
        `Callable`:
            The imported function.
    """
    # Check if it's a file path (contains .py:).
    if ".py:" in function_path:
        file_path, func_name = function_path.rsplit(":", 1)
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file_path}' is not found.")
        
        # Add the directory to sys.path temporarily.
        module_dir = str(file_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Import the module.
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from '{file_path}'.")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function.
        if not hasattr(module, func_name):
            raise AttributeError(f"Function '{func_name}' is not found in '{file_path}'.")
        
        return getattr(module, func_name)
    else:
        # Standard module import (e.g., "package.module.function").
        module_path, func_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        
        if not hasattr(module, func_name):
            raise AttributeError(f"Function '{func_name}' is not found in module '{module_path}'.")
        
        return getattr(module, func_name)


def download_model(
    repo_id: str,
    parent_dir: str | Path,
    repo_type: Literal["model", "dataset", "space"] | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    force: bool = False,
) -> None:
    """Download a single repository from Hugging Face.

    Args:
        repo_id (`str`):
            Repository ID in the format "organization/model-name" 
            (e.g., "Qwen/Qwen3-32B").
        parent_dir (`str | Path`):
            Parent directory where the repository will be downloaded.
            The model will be saved to `parent_dir/model-name/`.
        repo_type (`Literal["model", "dataset", "space"] | None`, optional):
            Type of repository. `None` or `"model"` for models.
        revision (`str | None`, optional):
            Specific revision (branch, tag, or commit) to download.
        allow_patterns (`list[str] | None`, optional):
            List of file patterns to download. If it is not provided, all files are downloaded.
        force (`bool`, defaults to `False`):
            If `True`, it will download even if the repository directory already exists.
            If `False`, it will skip the download if the repository directory already exists.
    """
    parent_dir = Path(parent_dir)
    repo_name = repo_id.split("/")[-1]
    repo_dir = parent_dir / repo_name

    # Check if the repository already exists.
    if repo_dir.exists() and not force:
        print(f"The repository '{repo_name}' already exists at '{repo_dir}'.")
        return

    if repo_dir.exists() and force:
        print(f"Re-download the repository '{repo_name}'.")

    # Download the repository.
    print(f"Download the repository '{repo_id}' to '{repo_dir}'.")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "`huggingface_hub` is not installed. "
            "Please install it with `pip install huggingface-hub`"
        ) from e
        
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(repo_dir),
        repo_type=repo_type,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    
    print(f"The repository '{repo_name}' is downloaded to '{repo_dir}' successfully.")


def download_models(
    repo_ids: str | list[str] | dict[str, dict],
    parent_dir: str | Path,
    repo_type: Literal["model", "dataset", "space"] | None = None,
    force: bool = False,
) -> None:
    """Download multiple repositories from Hugging Face.

    Args:
        repo_ids (`str | list[str] | dict[str, dict]`):
            Single repo ID string, list of repository IDs, or dict mapping 
            repo IDs to repository-specific configurations.
            
            Examples::

                repo_ids_1 = "Qwen/Qwen3-32B"`
                repo_ids_2 = ["Qwen/Qwen3-32B", "Qwen/Qwen3-4B"]
                repo_ids_3 = {"Qwen/Qwen3-32B": {"revision": "main", "force": True}}

        parent_dir (`str | Path`):
            Parent directory where repositories will be downloaded.
        repo_type (`Literal["model", "dataset", "space"] | None`, optional):
            The default repository type for all repositories. It can be overridden per repository.
        force (`bool`, defaults to `False`):
            The default force flag. It can be overridden per repository.
    """
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Normalize input to dict format.
    if isinstance(repo_ids, str):
        repo_ids = {repo_ids: {}}
    elif isinstance(repo_ids, list):
        repo_ids = {repo_id: {} for repo_id in repo_ids}

    # Download each repository.
    for repo_id, config in repo_ids.items():
        # Merge default and per-repository config.
        repo_config = {
            "repo_id": repo_id,
            "parent_dir": parent_dir,
            "repo_type": config.get("repo_type", repo_type),
            "revision": config.get("revision"),
            "allow_patterns": config.get("allow_patterns"),
            "force": config.get("force", force),
        }

        try:
            download_model(**repo_config)
        except Exception as e:
            print(f"Failed to download the repository '{repo_id}': {e}")
            raise

    print(f"All repositories are downloaded to '{parent_dir}' successfully.")