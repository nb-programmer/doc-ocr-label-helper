import logging
import urllib.parse
from itertools import islice
from pathlib import Path

import pandas as pd
import pymupdf
import pytesseract
from fastapi import Depends, Request
from PIL import Image

from ...config import (
    AppSettings,
    DatasetSettings,
    get_app_settings,
    get_dataset_settings,
)
from .depends import get_dataset_store_path
from .utils import convert_to_layout_studio_tasks, tesseract_to_tree

LOG = logging.getLogger(__name__)


class DatasetProcessService:
    def __init__(
        self,
        req: Request,
        dataset_settings: DatasetSettings = Depends(get_dataset_settings),
        dataset_path: Path = Depends(get_dataset_store_path),
    ):
        self._ds_settings = dataset_settings
        self._ds_store_path = dataset_path
        self._req = req

    def construct_serve_url(self, file: Path):
        static_file_path = urllib.parse.quote(file.name)
        return self._req.url_for("dataset_files", path=static_file_path)

    def image_ocr(self, file: Path) -> tuple[Path, tuple[int, int], pd.DataFrame]:
        with Image.open(file.absolute()) as image:
            return (
                file,
                image.size,
                pytesseract.image_to_data(
                    image,
                    output_type=pytesseract.Output.DATAFRAME,
                )
            )

    def ocr_to_tasks(self, file: Path, image_size: tuple[int, int], ocr_output: pd.DataFrame):
        tree = tesseract_to_tree(ocr_output)
        serve_url = self.construct_serve_url(file)
        return convert_to_layout_studio_tasks(tree, image_size, serve_url)

    def image_process(self, file: Path):
        ocr_result = self.image_ocr(file)
        return self.ocr_to_tasks(*ocr_result)

    def document_save_as_images(
        self,
        doc: pymupdf.Document,
        filename: str,
        limit_pages: int | None = None,
        save_file_extension: str = ".png",
        save_filename_format: str | None = None,
    ):
        if save_filename_format is None:
            save_filename_format = "{filename_stem}-page{page_number:02d}"

        for page in islice(doc, limit_pages):
            dest = self._ds_store_path.joinpath(filename).with_suffix(save_file_extension)
            dest = dest.with_stem(
                save_filename_format.format(
                    filename_stem=dest.stem,
                    page_number=page.number,
                )
            )

            if dest.exists():
                LOG.debug("Skipping `%s` as it already exists.", dest.name)
            else:
                dpi = self._ds_settings.image_render_dpi
                LOG.info("Saving page image `%s` at DPI %d", dest.name, dpi)
                pix: pymupdf.Pixmap = page.get_pixmap(dpi=dpi)
                pix.save(dest)

            yield dest

    def process_document(
        self,
        file_content: bytes,
        filename: str,
        limit_pages: int | None,
    ):
        with pymupdf.Document(stream=file_content) as doc:
            image_files_gen = self.document_save_as_images(
                doc,
                filename,
                limit_pages=limit_pages,
            )

            yield from map(self.image_process, image_files_gen)
