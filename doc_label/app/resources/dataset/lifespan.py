from contextlib import asynccontextmanager

from fastapi import FastAPI

from ...config import AppDirectories
from .depends import dataset_cache_path, dataset_export_path, dataset_store_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Dataset resource lifespan handler"""
    # Create directories used by the app if they don't exist
    app_dir_settings = AppDirectories()

    # Dataset store directory
    dataset_store_path(app_dir_settings).mkdir(parents=True, exist_ok=True)

    # Dataset cache directory
    dataset_cache_path(app_dir_settings).mkdir(parents=True, exist_ok=True)

    # Dataset export directory
    dataset_export_path(app_dir_settings).mkdir(parents=True, exist_ok=True)

    yield
