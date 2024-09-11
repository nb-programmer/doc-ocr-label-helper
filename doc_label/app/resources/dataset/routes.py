from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from ...config.models import AppDirectories
from .depends import dataset_store_path
from .views import convert_results_to_hf, create_document_ocr_tasks


def init_routes():
    router = APIRouter(prefix="/dataset", tags=["dataset"])

    router.add_api_route("/create_document_ocr_tasks", create_document_ocr_tasks, methods={"POST"})
    router.add_api_route("/convert_results_to_hf", convert_results_to_hf, methods={"POST"})

    return router


def init_app(app: FastAPI):
    app.include_router(init_routes())

    dataset_dir = dataset_store_path(AppDirectories())

    app.mount("/files/dataset", StaticFiles(directory=dataset_dir, check_dir=False), name="dataset_files")
