import os
from pathlib import Path
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
import pytesseract
import unidecode
from lxml import etree
from numpy import average
from PIL import Image as PILImage
from shapely.geometry import Polygon
from tqdm import tqdm

LABELS = []

label2id = {k: i for i, k in enumerate(LABELS)}


class LineDict(dict):
    @property
    def score(self):
        if len(self["word_data"]) == 0:
            return 0.0
        return float(
            average(list(map(lambda x: x["conf"], self["word_data"].values())))
        )

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
        return " ".join(
            filter(None, map(lambda x: x["text"], self["word_data"].values()))
        ).strip()


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


def custom_data(label_studio_data: list[dict]):
    document_data = dict()
    document_data["file_src"] = []
    document_data["file_name"] = []
    document_data["labelled_bbox"] = []

    # Iterate through tasks
    for row in label_studio_data:
        file_src = row["ocr"]
        file_name = os.path.basename(urlparse(file_src).path)

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

        document_data["file_src"].append(file_src)
        document_data["file_name"].append(file_name)
        document_data["labelled_bbox"].append(label_list)

    return pd.DataFrame(document_data)


def dataframe_to_dataset_hocr(custom_dataset: pd.DataFrame):
    final_list = []

    for i, row in tqdm(custom_dataset.iterrows(), total=custom_dataset.shape[0]):
        custom_label_text = {}
        word_list = []
        ner_tags_list = []
        bboxes_list = []

        custom_label_text["id"] = i
        custom_label_text["file_name"] = str(row["image_path"])

        image = Path(row["image_path"])
        label_coord_list = row["labelled_bbox"]

        hocr_save_path = image.parent.joinpath("layoutlmv3_hocr_output/")
        hocr_save_path.mkdir(exist_ok=True)

        for label_coord in label_coord_list:
            (x1, y1, x2, y2) = label_coord[1]
            box1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            label = label_coord[0][0]
            hocr_file = hocr_save_path.joinpath(image.name).with_suffix(".hocr")
            hocr_base_name = hocr_file.with_suffix("")

            # Perform HOCR
            pytesseract.pytesseract.run_tesseract(
                str(image),
                str(hocr_base_name),
                extension="box",
                lang=None,
                config="hocr",
            )

            hocr_df = hocr_to_dataframe(hocr_file)

            for _, word_obj in hocr_df.iterrows():
                coords = word_obj["coords"]
                x1df, y1df, x2df, y2df = coords

                box2 = [[x1df, y1df], [x2df, y1df], [x2df, y2df], [x1df, y2df]]

                word = word_obj["word"]
                word = unidecode.unidecode(word)

                overlap_perc = calculate_iou(box1, box2)

                if overlap_perc > 0.80:
                    if word != "-":
                        word_list.append(word)
                        bboxes_list.append(coords)
                        label_id = label2id[label]
                        ner_tags_list.append(label_id)

                    custom_label_text["tokens"] = word_list
                    custom_label_text["bboxes"] = bboxes_list
                    custom_label_text["ner_tags"] = ner_tags_list

        final_list.append(custom_label_text)

    return final_list
