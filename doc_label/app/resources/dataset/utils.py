import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
import pytesseract
import unidecode
from lxml import etree
from numpy import average
from shapely.geometry import Polygon
from tqdm.auto import tqdm

LOG = logging.getLogger(__name__)

SKIP_EMPTY = False

GetImagePathCallback = Callable[[str], Awaitable[str]]
Label2ID = Callable[[str], int | None]


class LineDict(dict):
    @property
    def score(self):
        if len(self["word_data"]) == 0:
            return 0.0
        return float(average(list(map(lambda x: x["conf"], self["word_data"].values()))))

    @property
    def box(self):
        return {
            "x": self["line_data"]["left"],
            "y": self["line_data"]["top"],
            "width": self["line_data"]["width"],
            "height": self["line_data"]["height"],
            "rotation": 0,
        }

    @property
    def text(self):
        return " ".join(filter(None, map(lambda x: str(x["text"]), self["word_data"].values()))).strip()


def convert_to_layout_studio_tasks(ocr_tree: dict[tuple, LineDict], image_size: tuple[int, int], serve_url: str):
    results = []
    all_scores = []

    image_width, image_height = image_size

    for row in ocr_tree.values():
        bbox = row.box
        bbox.update(
            {
                "x": 100 * bbox["x"] / image_width,
                "y": 100 * bbox["y"] / image_height,
                "width": 100 * bbox["width"] / image_width,
                "height": 100 * bbox["height"] / image_height,
            }
        )

        text = row.text
        score = row.score

        if not text:
            continue

        region_id = str(uuid4())[:10]

        bbox_result = {
            "id": region_id,
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectangle",
            "value": bbox,
        }
        transcription_result = {
            "id": region_id,
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "value": dict(text=[text], **bbox),
            "score": score,
        }
        results.extend([bbox_result, transcription_result])
        all_scores.append(score)

    average_score = sum(all_scores) / len(all_scores) if all_scores else 0

    return {
        "data": {
            "ocr": serve_url,
        },
        "predictions": [
            {
                "result": results,
                "score": average_score,
            }
        ],
    }


def tesseract_to_tree(data: pd.DataFrame):
    lines = {}
    for _, row in data.iterrows():
        line_id = (row.page_num, row.block_num, row.par_num, row.line_num)
        if row.line_num == 0:
            continue

        lines.setdefault(line_id, LineDict())

        match row.level:
            case 5:
                lines[line_id].setdefault("word_data", {}).update({row.word_num: row})
            case 4:
                lines[line_id].update(line_data=row)

    return lines


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    # print(poly_1,poly_2)
    # iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    iou = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area, poly_2.area)
    return iou / min_area


def hocr_to_dataframe(fp):
    doc = etree.parse(fp)
    words = []
    wordConf = []
    coords_list = []
    for path in doc.xpath("//*"):
        if "ocrx_word" in path.values():
            coord_text = path.values()[2].split(";")[0].split(" ")[1:]
            word_coord = list(map(int, coord_text))  # x1, y1, x2, y2
            conf = [x for x in path.values() if "x_wconf" in x][0]
            wordConf.append(int(conf.split("x_wconf ")[1]))
            words.append(path.text)
            coords_list.append(word_coord)

    dfReturn = pd.DataFrame(
        {
            "word": words,
            "coords": coords_list,
            "confidence": wordConf,
        }
    )

    return dfReturn


def url_to_filename(src: str) -> str:
    return PurePosixPath(urlparse(src).path).name


async def default_ocr_image_path(src: str) -> str:
    return str(Path(url_to_filename(src)).absolute())


async def parse_custom_dataset_to_df(
    label_studio_data: list[dict], get_image_path_callback: GetImagePathCallback = default_ocr_image_path
):
    document_data = []

    # Iterate through tasks
    for row in label_studio_data:
        # Contains URL to the image
        file_src = row["ocr"]

        # Just the filename
        file_name = url_to_filename(file_src)

        # Get path to image (or download if needed)
        image_path = await get_image_path_callback(file_src)

        label_list = []

        # Go through all labelled boxes
        for label_ in row.get("label", []):
            x, y, w, h = (
                label_["x"],
                label_["y"],
                label_["width"],
                label_["height"],
            )
            original_w, original_h = label_["original_width"], label_["original_height"]

            x1 = int((x * original_w) / 100)
            y1 = int((y * original_h) / 100)
            x2 = x1 + int(original_w * w / 100)
            y2 = y1 + int(original_h * h / 100)

            labels = label_["labels"]

            label_list.append((labels, (x1, y1, x2, y2), original_h, original_w))

        # Add collected data to df
        document_data.append(
            {
                "file_src": file_src,
                "file_name": file_name,
                "image_path": image_path,
                "labelled_bbox": label_list,
            }
        )

    return pd.DataFrame(document_data)


def dataframe_to_dataset_hocr(custom_dataset: pd.DataFrame, label2id: Label2ID, hocr_save_path: Path):
    final_list = []

    LOG.info("Running hOCR for the following dataset (first 5 rows):\n%s", custom_dataset.head(5))

    for i, row in tqdm(
        custom_dataset.iterrows(), total=custom_dataset.shape[0], unit="item", desc="Task hOCR", colour="blue"
    ):
        custom_label_text = {}
        word_list = []
        ner_tags_list = []
        bboxes_list = []

        image_file_path: str | None = row["image_path"]
        if image_file_path is None or image_file_path == "":
            LOG.warning("Image for task `%s` was null (not found in dataset?). Skipping.", i)
            continue

        image_path = str(image_file_path)
        image_filename = os.path.basename(image_path)

        custom_label_text["id"] = i
        custom_label_text["image"] = image_path
        custom_label_text["tokens"] = []
        custom_label_text["bboxes"] = []
        custom_label_text["ner_tags"] = []

        label_coord_list = row["labelled_bbox"]

        if SKIP_EMPTY and len(label_coord_list) == 0:
            LOG.info("Skipping task `%s` as there are no labels.", i)
            continue

        ### OCR the image with HOCR output ###

        # Location to store the hOCR results
        hocr_file = hocr_save_path.joinpath(image_filename).with_suffix(".hocr")

        if not hocr_file.exists():
            # PyTesseract needs the filename without extension, so remove it.
            hocr_base_name = str(hocr_file.with_suffix(""))

            # Perform HOCR
            pytesseract.pytesseract.run_tesseract(
                image_path,
                hocr_base_name,
                extension="box",
                lang=None,
                config="hocr",
            )

        # Convert the saved HOCR file to DF
        hocr_df = hocr_to_dataframe(hocr_file)

        # Apply to each box labelled in the task
        for label_coord in tqdm(label_coord_list, unit="box", desc="Box hOCR", colour="green", leave=False):
            (x1, y1, x2, y2) = label_coord[1]
            box1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            label = label_coord[0][0]

            for _, word_obj in hocr_df.iterrows():
                coords = word_obj["coords"]
                x1df, y1df, x2df, y2df = coords

                box2 = [[x1df, y1df], [x2df, y1df], [x2df, y2df], [x1df, y2df]]

                word = word_obj["word"]
                word = unidecode.unidecode(word)

                overlap_perc = calculate_iou(box1, box2)

                if overlap_perc > 0.80:
                    if word != "-":
                        # Get label id
                        label_id = label2id(label)

                        word_list.append(word)
                        bboxes_list.append(coords)
                        ner_tags_list.append(label_id)

        # Add data to the row
        custom_label_text["tokens"] = word_list
        custom_label_text["bboxes"] = bboxes_list
        custom_label_text["ner_tags"] = ner_tags_list

        final_list.append(custom_label_text)

    return final_list


@contextmanager
def random_context():
    yield (old_state := random.getstate())
    random.setstate(old_state)
