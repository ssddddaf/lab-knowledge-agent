import hashlib
from pathlib import Path

from qdrant_client import QdrantClient, models

from .config import Settings


class VectorIndex:
    def __init__(self, config: Settings):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url, timeout=30)
        self._text_model = None
        self._vision = None
        self._vision_processor = None

    def _ensure_collection(self, name: str, size: int, multivector: bool = False):
        if self.client.collection_exists(name):
            return
        params = models.VectorParams(size=size, distance=models.Distance.COSINE)
        if multivector:
            params.multivector_config = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
        self.client.create_collection(collection_name=name, vectors_config=params)

    def _dense(self, text: str):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer("BAAI/bge-m3")
        return self._text_model.encode(text, normalize_embeddings=True).tolist()

    def index_text(self, blocks: list[dict]):
        self._ensure_collection("text", 1024)
        points=[]
        for b in blocks:
            points.append(models.PointStruct(
                id=self._point_id(b["id"]), vector=self._dense(b["text"]),
                payload={"block_id":b["id"],"document_id":b["document_id"],"page_number":b["page_number"],"text":b["text"],"bbox":b.get("bbox")},
            ))
        if points:
            self.client.upsert("text", points=points, wait=True)

    @staticmethod
    def _point_id(value: str) -> int:
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") & ((1 << 63)-1)

    def _load_vision(self):
        if self._vision is None:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            if not torch.cuda.is_available():
                raise RuntimeError("ColQwen2 requires CUDA GPU")
            self._vision = ColQwen2.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16, device_map="cuda").eval()
            self._vision_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    def _image_vectors(self, image_path: Path):
        import torch
        from PIL import Image
        self._load_vision()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            batch = self._vision_processor.process_images([image]).to(self._vision.device)
        with torch.no_grad():
            result = self._vision(**batch)
        return result[0].float().cpu().tolist()

    def _query_vectors(self, text: str):
        import torch
        self._load_vision()
        batch = self._vision_processor.process_queries([text]).to(self._vision.device)
        with torch.no_grad():
            result = self._vision(**batch)
        return result[0].float().cpu().tolist()

    def index_page(self, document_id: str, page_number: int, image_path: Path):
        self._ensure_collection("pages_visual", 128, multivector=True)
        self.client.upsert("pages_visual", points=[models.PointStruct(
            id=self._point_id(f"{document_id}:{page_number}"),
            vector=self._image_vectors(image_path),
            payload={"document_id":document_id,"page_number":page_number},
        )], wait=True)

    def search_text(self, question: str, document_ids: list[str], limit: int = 20):
        self._ensure_collection("text",1024)
        filt = models.Filter(must=[models.FieldCondition(key="document_id",match=models.MatchAny(any=document_ids))]) if document_ids else None
        return self.client.query_points("text",query=self._dense(question),query_filter=filt,limit=limit,with_payload=True).points

    def search_visual(self, question: str, document_ids: list[str], limit: int = 10):
        self._ensure_collection("pages_visual",128,multivector=True)
        filt = models.Filter(must=[models.FieldCondition(key="document_id",match=models.MatchAny(any=document_ids))]) if document_ids else None
        return self.client.query_points("pages_visual",query=self._query_vectors(question),query_filter=filt,limit=limit,with_payload=True).points
