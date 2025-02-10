# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Header, Response
from typing import Union, Annotated

from app.schemas import OCRBlock, OCRIn, OCRLine, OCROut, OCRPage, OCRWord
from app.utils import get_documents, resolve_geometry
from app.vision import init_predictor

from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement
import doctr

router = APIRouter()

hocr_types ={ "text/xml", "application/xml", "text/vnd.hocr+html", "text/vnd.hocr+xml", "text/html" }

@router.post("/", response_model=Union[str,list[OCROut]], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(request: OCRIn = Depends(), files: list[UploadFile] = [File(...)], accept: Annotated[str |None, Header()]=None):
    """Runs docTR OCR model to analyze the input image"""
    try:
        # generator object to list
        content, filenames = await get_documents(files)
        predictor = init_predictor(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    out = predictor(content)

    if accept in hocr_types:
        if len(out.pages) > 0:
            page0 = out.pages[0]
            language = page0.language if "language" in page0.language.keys() else "en"
        else:
            language = "en"
        # Create the XML root element
        doc_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(doc_hocr, "head")
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(doc_hocr, "body")
        for page in out.pages:
            page.export_page_as_xml(body)

        return Response(content=ET.tostring(doc_hocr, encoding="utf-8", method="xml"), media_type="text/xml")
    else:
        results = [
                OCROut(
                    name=filenames[i],
                    orientation=page.orientation,
                    language=page.language,
                    dimensions=page.dimensions,
                    items=[
                        OCRPage(
                            blocks=[
                                OCRBlock(
                                    geometry=resolve_geometry(block.geometry),
                                    objectness_score=round(block.objectness_score, 2),
                                    lines=[
                                        OCRLine(
                                            geometry=resolve_geometry(line.geometry),
                                            objectness_score=round(line.objectness_score, 2),
                                            words=[
                                                OCRWord(
                                                    value=word.value,
                                                    geometry=resolve_geometry(word.geometry),
                                                    objectness_score=round(word.objectness_score, 2),
                                                    confidence=round(word.confidence, 2),
                                                    crop_orientation=word.crop_orientation,
                                                )
                                        for word in line.words
                                            ],
                                        )
                                for line in block.lines
                                    ],
                                )
                        for block in page.blocks
                            ]
                        )
                    ],
                )
        for i, page in enumerate(out.pages)
            ]
        return results
