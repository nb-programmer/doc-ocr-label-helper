import json
import logging
import random
import shutil
import tempfile
import urllib.parse
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Annotated
from uuid import uuid4

import datasets
import numpy as np
import yaml
from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .depends import (
    get_dataset_cache_path,
    get_dataset_export_path,
    get_dataset_store_path,
)
from .services import DatasetProcessService
from .utils import (
    async_random_context,
    dataframe_to_dataset_hocr,
    dataset_normalize_bboxes,
    parse_custom_dataset_to_df,
)

DATASET_README_FORMAT = """---
{config_yaml}
---
"""

TRAIN_SET_FILENAME = "train.jsonl"
TEST_SET_FILENAME = "test.jsonl"
README_FILENAME = "README.md"
MAX_SHARD_SIZE = "64MB"
DS_ARCHIVE_FORMAT = "zip"


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
    dataset_name: str = Form(default="dataset"),
    dataset_path: Path = Depends(get_dataset_store_path),
    cache_path: Path = Depends(get_dataset_cache_path),
    export_path: Path = Depends(get_dataset_export_path),
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

    # In case we receive an empty dataset name, revert to default
    if dataset_name == "":
        dataset_name = "dataset"

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

    # Re-scale bounding boxes to 0-1000 range
    dataset_normalize_bboxes(final_dataset_list)

    # Figure out rows with boxes and without
    data_with_boxes = list(filter(lambda r: len(r["bboxes"]) > 0, final_dataset_list))
    data_no_boxes = list(filter(lambda r: len(r["bboxes"]) == 0, final_dataset_list))

    # Oversampling and shuffle
    async with async_random_context():
        random.seed(random_seed)

        # Unbalanced dataset oversampling
        if len(data_with_boxes) != len(data_no_boxes):
            count_diff = abs(len(data_with_boxes) - len(data_no_boxes))
            least_class = data_with_boxes if len(data_with_boxes) < len(data_no_boxes) else data_no_boxes
            oversamples = np.random.choice(least_class, size=count_diff, replace=True)
            final_dataset_list.extend(oversamples)

        # Shuffle dataset
        random.shuffle(final_dataset_list)

    # Test/train split

    train_count = int(len(final_dataset_list) * train_split_percent)
    test_count = len(final_dataset_list) - train_count

    train_set = final_dataset_list[:train_count]
    test_set = final_dataset_list[-test_count:]

    assert len(train_set) == train_count
    assert len(test_set) == test_count

    LOG.info(
        "Split dataset of %d elements into train/test: %d/%d.", len(final_dataset_list), len(train_set), len(test_set)
    )

    # Export to HuggingFace Datasets format

    # TODO: aiofiles
    with tempfile.TemporaryDirectory(prefix="dataset-", dir=cache_path) as tmpdir:
        tmpdir = Path(tmpdir)

        rand_id = uuid4()

        dataset_load_name = "%s-%s" % (dataset_name, rand_id)

        dataset_path = tmpdir / dataset_load_name
        dataset_path.mkdir(exist_ok=True)

        ds_save_path = tmpdir / ("%s.hf/" % dataset_name)
        ds_archive_export_file = export_path / tmpdir.name / dataset_name

        dataset_config = {
            "configs": [
                {
                    "config_name": "default",
                    "data_files": [
                        {
                            "split": "train",
                            "path": TRAIN_SET_FILENAME,
                        },
                        {
                            "split": "test",
                            "path": TEST_SET_FILENAME,
                        },
                    ],
                }
            ],
            "dataset_info": {
                "features": [
                    {"name": "id", "dtype": "string"},
                    {"name": "filename", "dtype": "string"},
                    {"name": "tokens", "sequence": {"dtype": "string"}},
                    {"name": "bboxes", "sequence": {"sequence": {"dtype": "int64"}}},
                    {
                        "name": "ner_tags",
                        "sequence": {"dtype": {"class_label": {"names": labels_list}}},
                    },
                    {"name": "image", "dtype": "image"},
                ]
            },
        }

        # Write training set as JSONL format
        with open(dataset_path / TRAIN_SET_FILENAME, "w") as f:
            for detail in train_set:
                f.write(json.dumps(detail))
                f.write("\n")

        # Write testing set as JSONL format
        with open(dataset_path / TEST_SET_FILENAME, "w") as f:
            for detail in test_set:
                f.write(json.dumps(detail))
                f.write("\n")

        # Write README file containing dataset config and description
        with open(dataset_path / README_FILENAME, "w") as f:
            f.write(
                DATASET_README_FORMAT.format(
                    config_yaml=yaml.safe_dump(dataset_config),
                )
            )

        # Create dataset from folder and save it to disk
        datasets.load_dataset(str(dataset_path)).save_to_disk(ds_save_path, max_shard_size=MAX_SHARD_SIZE)

        # Compress to zip
        archive_path = Path(shutil.make_archive(ds_archive_export_file, DS_ARCHIVE_FORMAT, ds_save_path))

        return FileResponse(archive_path, filename=archive_path.name)
