import json
import logging
import random
import urllib.parse
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Annotated

import numpy as np
from fastapi import Depends, File, Form, HTTPException, UploadFile, status

from .depends import get_dataset_cache_path, get_dataset_store_path
from .services import DatasetProcessService
from .utils import dataframe_to_dataset_hocr, parse_custom_dataset_to_df, random_context

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
    train_split_percent: float = Form(default=0.8, ge=0.0, le=1.0),
    random_seed: str | None = Form(default=""),
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

    # Empty RNG seed should be parsed as `None` to indicate random seed.
    if isinstance(random_seed, str) and random_seed == "":
        random_seed = None

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

    data_with_boxes = list(filter(lambda r: len(r["bboxes"]) > 0, final_dataset_list))
    data_no_boxes = list(filter(lambda r: len(r["bboxes"]) == 0, final_dataset_list))

    with random_context():
        random.seed(random_seed)

        # Unbalanced dataset oversampling
        if len(data_with_boxes) != len(data_no_boxes):
            count_diff = abs(len(data_with_boxes) - len(data_no_boxes))
            least_class = data_with_boxes if len(data_with_boxes) < len(data_no_boxes) else data_no_boxes
            oversamples = np.random.choice(least_class, size=count_diff, replace=True)
            final_dataset_list.extend(oversamples)

        # Shuffle dataset
        random.shuffle(final_dataset_list)

    train_count = int(len(final_dataset_list) * train_split_percent)
    test_count = len(final_dataset_list) - train_count

    train_set = final_dataset_list[:train_count]
    test_set = final_dataset_list[-test_count:]

    assert len(train_set) == train_count
    assert len(test_set) == test_count

    breakpoint()
