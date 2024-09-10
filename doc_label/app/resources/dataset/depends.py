from pathlib import Path

from fastapi import Depends

from ...config import AppDirectories, get_app_dirs


def dataset_store_path(app_dirs: AppDirectories):
    return app_dirs.data_dir.joinpath("store/")


def get_dataset_store_path(
    app_dirs: AppDirectories = Depends(get_app_dirs),
) -> Path:
    return dataset_store_path(app_dirs)
