import hashlib
import json
import re
from pathlib import Path

import fitz

from .config import Settings
from .graph_store import GraphStore
from .llm import LLM
from .models import Extraction
from .store import Store
from .vectors import VectorIndex


def chunk_text(text: str, max_words: int = 500, overlap: int = 70):
    words = text.split()
    if not words:
        return []
    result=[]
    step=max_words-overlap
    for start in range(0,len(words),step):
        result.append(" ".join(words[start:start+max_words]))
        if start+max_words>=len(words):
            break
    return result


class Ingestor:
    def __init__(self, config: Settings, store: Store):
        self.config=config
        self.store=store

    def register(self, filename: str, data: bytes) -> str:
        if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
            raise ValueError("Only PDF files are supported")
        document_id=hashlib.sha256(data).hexdigest()
        folder=self.config.data_dir / "documents" / document_id
        folder.mkdir(parents=True,exist_ok=True)
        source=folder / "source.pdf"
        if not source.exists():
            source.write_bytes(data)
        self.store.add_document(document_id,Path(filename).name,str(source))
        return document_id

    def _ocr_blocks(self, image: Path, document_id: str, page_number: int):
        if not self.config.enable_ocr:
            return []
        from paddleocr import PPStructureV3
        if not hasattr(self,"_ocr"):
            self._ocr=PPStructureV3()
        result=list(self._ocr.predict(input=str(image)))
        if not result:
            return []
        raw=result[0].json if hasattr(result[0],"json") else result[0]
        if callable(raw):
            raw=raw()
        raw=raw.get("res",raw)
        output=[]
        for i,block in enumerate(raw.get("parsing_res_list",[])):
            text=block.get("block_content","")
            if not text.strip():
                continue
            bbox=block.get("block_bbox")
            bbox=[float(x)/1.5 for x in bbox] if bbox and len(bbox)==4 else None
            output.append({"id":f"{document_id}:{page_number}:ocr:{i}","document_id":document_id,
                           "page_number":page_number,"kind":block.get("block_label","text"),
                           "text":text,"bbox":bbox})
        return output

    def _native_blocks(self, page, document_id: str, number: int):
        output=[]
        for i,b in enumerate(page.get_text("blocks")):
            if len(b)<5 or not str(b[4]).strip():
                continue
            text=str(b[4]).strip()
            for j,part in enumerate(chunk_text(text)):
                output.append({"id":f"{document_id}:{number}:native:{i}:{j}","document_id":document_id,
                               "page_number":number,"kind":"text","text":part,"bbox":list(b[:4])})
        return output

    def process(self, document_id: str) -> dict:
        doc=self.store.document(document_id)
        if not doc:
            raise ValueError("Unknown document")
        self.store.set_document_status(document_id,"processing")
        page_count=0
        try:
            pdf=fitz.open(doc["path"])
            page_count=len(pdf)
            if page_count>1000:
                raise ValueError("PDF exceeds 1000 pages")
            vector=VectorIndex(self.config) if (self.config.enable_dense or self.config.enable_vision) else None
            graph=None
            try:
                graph=GraphStore(self.config) if self.config.neo4j_password else None
            except Exception:
                graph=None
            llm=LLM(self.config)
            for number,page in enumerate(pdf,1):
                image=self.config.data_dir / "documents" / document_id / f"page-{number}.png"
                if not image.exists():
                    page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(image)
                native=self._native_blocks(page,document_id,number)
                ocr=self._ocr_blocks(image,document_id,number) if (not native or self.config.enable_ocr) else []
                blocks=native+ocr
                page_text="\n".join(b["text"] for b in blocks)
                self.store.add_page(document_id,number,str(image),page_text)
                self.store.add_blocks(blocks)
                if blocks and self.config.enable_dense and vector:
                    vector.index_text(blocks)
                if self.config.enable_vision and vector:
                    vector.index_page(document_id,number,image)
                if graph and llm.enabled and page_text.strip():
                    try:
                        extraction=llm.json(Extraction,"Extract only explicit scientific entities and relations. Every relation evidence_id must be an exact supplied block ID. Return empty lists when uncertain.",
                                            json.dumps({"blocks":[{"id":b["id"],"text":b["text"][:1500]} for b in blocks[:12]]},ensure_ascii=False))
                        graph.add_extraction(extraction,document_id,number)
                    except Exception:
                        # Graph extraction is an enrichment; document search remains available.
                        pass
            pdf.close()
            if graph:
                graph.close()
            self.store.set_document_status(document_id,"ready",pages=page_count)
            return {"document_id":document_id,"status":"ready","page_count":page_count}
        except Exception as exc:
            self.store.set_document_status(document_id,"failed",str(exc),page_count)
            raise
