import json
import logging
import urllib.parse
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import Depends, File, Form, HTTPException, UploadFile, status

from .depends import get_dataset_cache_path, get_dataset_store_path
from .services import DatasetProcessService
from .utils import dataframe_to_dataset_hocr, parse_custom_dataset_to_df

LOG = logging.getLogger(__name__)

DocumentUploads = Annotated[list[UploadFile], File()]
JSONUpload = Annotated[UploadFile, File()]


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


async def _dataset_path_or_download(src: str, dataset_path: Path, cache_path: Path) -> str:
    # Extract file name portion, path join with dataset directory
    image_path = dataset_path / (PurePosixPath(urllib.parse.unquote(urllib.parse.urlparse(src).path)).name)

    # If it exists, we got it
    if image_path.exists():
        LOG.debug("`%s` exists in dataset.", src)
        return str(image_path)

    # No, we need to fetch it.
    LOG.info("`%s` not present in dataset. Need to fetch `%s`...", image_path.name, src)
    # TODO
    # return ...


def validate_annotation_json(obj) -> bool:
    return isinstance(obj, list)


def validate_labels_list_json(obj) -> bool:
    return isinstance(obj, list)


async def convert_results_to_hf(
    json_file: JSONUpload,
    labels_file: JSONUpload,
    dataset_path: Path = Depends(get_dataset_store_path),
    cache_path: Path = Depends(get_dataset_cache_path),
):
    label_studio_data: list[dict] = json.load(json_file.file)
    labels_list: list[str] = json.load(labels_file.file)

    if not validate_annotation_json(label_studio_data):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Label studio file is not in a valid format. It should be a valid JSON file with array items.",
        )

    if not validate_labels_list_json(labels_list):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Labels list file is not in a valid format. It should be a valid JSON file with array items of label strings.",
        )

    label_id_map = {v: i for i, v in enumerate(labels_list)}

    def label2id(s):
        return label_id_map.get(s, -1)

    labelled_df = await parse_custom_dataset_to_df(
        label_studio_data,
        get_image_path_callback=partial(
            _dataset_path_or_download,
            dataset_path=dataset_path,
            cache_path=cache_path,
        ),
    )

    final_dataset_list = dataframe_to_dataset_hocr(labelled_df, label2id=label2id, hocr_save_path=cache_path)
    final_dataset_list

    breakpoint()
