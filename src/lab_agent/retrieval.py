import json
import re
from collections import defaultdict

from .config import Settings
from .graph_store import GraphStore
from .models import Evidence
from .store import Store
from .vectors import VectorIndex


def rrf(lists: list[list[Evidence]], limit: int = 6, k: int = 60) -> list[Evidence]:
    scores=defaultdict(float)
    chosen={}
    for ranked in lists:
        seen=set()
        for rank,e in enumerate(ranked,1):
            key=(e.document_id,e.page_number,e.id if e.channel=="project" else "")
            if key in seen:
                continue
            seen.add(key)
            scores[key]+=1/(k+rank)
            if key not in chosen or (e.bbox and not chosen[key].bbox):
                chosen[key]=e
    return [chosen[key].model_copy(update={"score":scores[key]}) for key in sorted(scores,key=scores.get,reverse=True)[:limit]]


class Retriever:
    def __init__(self, config: Settings, store: Store):
        self.config=config
        self.store=store
        self.degraded=[]

    def _evidence(self,b:dict, channel:str, score:float=0) -> Evidence:
        bbox=b.get("bbox")
        if isinstance(bbox,str):
            bbox=json.loads(bbox)
        doc=self.store.document(b["document_id"])
        return Evidence(id=b.get("id") or f'{b["document_id"]}:{b["page_number"]}',
                        document_id=b["document_id"],version_id=doc["version_id"] if doc else b["document_id"],
                        page_number=int(b["page_number"]),block_id=b.get("id"),bbox=bbox,
                        text=b.get("text","")[:3500],channel=channel,score=score)

    def search(self, question:str, document_ids:list[str]|None=None, project_id:str|None=None, limit:int=6) -> list[Evidence]:
        document_ids=document_ids or []
        channels=[]
        blocks=self.store.blocks(document_ids)
        tokens=re.findall(r"[\w.-]+",question.casefold())
        if blocks and tokens:
            from rank_bm25 import BM25Okapi
            corpus=[re.findall(r"[\w.-]+",b["text"].casefold()) for b in blocks]
            scores=BM25Okapi(corpus).get_scores(tokens)
            ordered=sorted(range(len(blocks)),key=lambda i:scores[i],reverse=True)[:20]
            channels.append([self._evidence(blocks[i],"bm25",float(scores[i])) for i in ordered if scores[i]>0])
        if self.config.enable_dense or self.config.enable_vision:
            try:
                vector=VectorIndex(self.config)
                if self.config.enable_dense:
                    points=vector.search_text(question,document_ids)
                    channels.append([self._evidence({"id":p.payload["block_id"],"document_id":p.payload["document_id"],
                        "page_number":p.payload["page_number"],"text":p.payload.get("text",""),"bbox":p.payload.get("bbox")},"dense",p.score) for p in points])
                if self.config.enable_vision:
                    points=vector.search_visual(question,document_ids)
                    visual=[]
                    for p in points:
                        page=self.store.page(p.payload["document_id"],p.payload["page_number"])
                        if page:
                            visual.append(self._evidence({"document_id":page["document_id"],"page_number":page["page_number"],"text":page["text"]},"visual",p.score))
                    channels.append(visual)
            except Exception as exc:
                if not self.config.allow_text_only:
                    raise
                self.degraded.append(f"vector_unavailable:{type(exc).__name__}")
        if not self.config.enable_dense:
            self.degraded.append("dense_disabled")
        if not self.config.enable_vision:
            self.degraded.append("vision_disabled")
        if self.config.neo4j_password:
            try:
                graph=GraphStore(self.config)
                terms=[t for t in tokens if len(t)>3][:5]
                related=graph.related_pages(terms)
                graph.close()
                rows=[]
                for hit in related:
                    if document_ids and hit["document_id"] not in document_ids:
                        continue
                    page=self.store.page(hit["document_id"],hit["page_number"])
                    if page:
                        rows.append(self._evidence({"document_id":page["document_id"],"page_number":page["page_number"],"text":hit.get("context") or page["text"]},"graph"))
                channels.append(rows)
            except Exception as exc:
                self.degraded.append(f"graph_unavailable:{type(exc).__name__}")
        if project_id:
            runs=self.store.lineage(project_id)
            matches=[r for r in runs if any(t in json.dumps(r,ensure_ascii=False).casefold() for t in tokens)]
            if matches:
                channels.append([Evidence(id=f'run:{project_id}:{r["run_id"]}',document_id=f'project:{project_id}',
                            version_id=r["run_id"],page_number=0,text=json.dumps(r,ensure_ascii=False),
                            channel="project",score=1.0) for r in matches[:10]])
        return rrf(channels,limit=limit)
