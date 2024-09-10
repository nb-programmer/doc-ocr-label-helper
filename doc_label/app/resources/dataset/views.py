import json
import logging
from typing import Annotated

from fastapi import Depends, File, Form, Request, UploadFile

from .services import DatasetProcessService

LOG = logging.getLogger(__name__)

DocumentUploads = Annotated[list[UploadFile], File()]
AnnotationJSONUpload = Annotated[UploadFile, File()]


async def create_document_ocr_tasks(
    files: DocumentUploads,
    limit_pages: int | None = Form(default=99, ge=1),
    dataset_process_service: DatasetProcessService = Depends(DatasetProcessService),
):
    all_ocr_tasks = []
    LOG.info("Processing %d documents...", len(files))

    for file in files:
        filename = file.filename
        stream = await file.read()

        all_ocr_tasks.extend(
            dataset_process_service.process_document(
                stream,
                filename,
                limit_pages,
            )
        )

    return all_ocr_tasks


async def convert_results_to_hf(json_file: AnnotationJSONUpload):
    data = json.load(json_file.file)  # noqa
    breakpoint()
