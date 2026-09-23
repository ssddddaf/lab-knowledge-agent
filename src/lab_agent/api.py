import asyncio
import json
import uuid
from pathlib import Path
from io import BytesIO

import fitz

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, Response

from .config import settings
from .graph_store import GraphStore
from .ingest import Ingestor
from .models import GraphQuery, QueryInput, RunInput
from .store import Store
from .workflow import Agent

config=settings()
store=Store(config.db_path)
app=FastAPI(title="Laboratory Knowledge Agent",version="0.1.0")


def ingest_job(job_id:str, document_id:str):
    store.set_job(job_id,document_id,"running")
    try:
        Ingestor(config,store).process(document_id)
        store.set_job(job_id,document_id,"done")
    except Exception as exc:
        store.set_job(job_id,document_id,"failed",str(exc))


def query_job(query_id:str, query:QueryInput):
    store.set_query(query_id,"running")
    try:
        answer=Agent(config,store).ask(query,query_id,on_node=lambda node:store.add_query_event(query_id,node))
        store.set_query(query_id,"done",answer.model_dump_json())
    except Exception as exc:
        store.set_query(query_id,"failed",error=str(exc))


@app.get("/health")
def health():
    return {"status":"ok","vision_enabled":config.enable_vision,"dense_enabled":config.enable_dense,"model_enabled":bool(config.chat_model and config.openai_api_key)}


@app.get("/documents")
def documents():
    return store.documents()


@app.post("/documents",status_code=202)
async def upload_document(background:BackgroundTasks,file:UploadFile=File(...)):
    data=await file.read()
    if len(data)>100*1024*1024:
        raise HTTPException(413,"PDF exceeds 100 MB")
    try:
        document_id=Ingestor(config,store).register(file.filename or "document.pdf",data)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    existing=store.document(document_id)
    job_id=str(uuid.uuid4())
    if existing and existing["status"]=="ready":
        store.set_job(job_id,document_id,"done")
    else:
        store.set_job(job_id,document_id,"queued")
        background.add_task(ingest_job,job_id,document_id)
    return {"job_id":job_id,"document_id":document_id}


@app.get("/jobs/{job_id}")
def job(job_id:str):
    result=store.job(job_id)
    if not result:
        raise HTTPException(404,"Job not found")
    return result


@app.get("/documents/{document_id}/pages/{page_number}")
def page(document_id:str,page_number:int):
    result=store.page(document_id,page_number)
    if not result:
        raise HTTPException(404,"Page not found")
    return FileResponse(Path(result["image_path"]),media_type="image/png")


@app.get("/documents/{document_id}/pages/{page_number}/metadata")
def page_metadata(document_id:str,page_number:int):
    result=store.page(document_id,page_number)
    if not result:
        raise HTTPException(404,"Page not found")
    blocks=[b for b in store.blocks([document_id]) if b["page_number"]==page_number]
    return {"document_id":document_id,"page_number":page_number,"blocks":blocks}


@app.get("/documents/{document_id}/pages/{page_number}/regions/{block_id:path}")
def page_region(document_id:str,page_number:int,block_id:str):
    doc=store.document(document_id)
    if not doc:
        raise HTTPException(404,"Document not found")
    blocks=[b for b in store.blocks([document_id]) if b["page_number"]==page_number and b["id"]==block_id]
    if not blocks or not blocks[0]["bbox"]:
        raise HTTPException(404,"Region not found")
    bbox=json.loads(blocks[0]["bbox"])
    if len(bbox)!=4:
        raise HTTPException(422,"Invalid region")
    pdf=fitz.open(doc["path"])
    try:
        region=fitz.Rect(bbox) & pdf[page_number-1].rect
        if region.is_empty:
            raise HTTPException(422,"Region outside page")
        image=pdf[page_number-1].get_pixmap(matrix=fitz.Matrix(2,2),clip=region,alpha=False).tobytes("png")
        return Response(image,media_type="image/png")
    finally:
        pdf.close()


@app.post("/queries",status_code=202)
def submit_query(query:QueryInput,background:BackgroundTasks):
    query_id=str(uuid.uuid4())
    store.set_query(query_id,"queued")
    background.add_task(query_job,query_id,query)
    return {"query_id":query_id}


@app.get("/queries/{query_id}")
def get_query(query_id:str):
    result=store.query(query_id)
    if not result:
        raise HTTPException(404,"Query not found")
    if result["result"]:
        result["result"]=json.loads(result["result"])
    result["events"]=store.query_events(query_id)
    return result


@app.get("/queries/{query_id}/events")
async def query_events(query_id:str):
    async def events():
        last=None
        seq=0
        for _ in range(120):
            result=store.query(query_id)
            if not result:
                yield 'event: error\ndata: {"error":"Query not found"}\n\n'
                return
            if result["status"]!=last or result["status"] in {"done","failed"}:
                payload={"status":result["status"],"error":result["error"]}
                if result["result"]:
                    payload["result"]=json.loads(result["result"])
                yield "event: status\ndata: "+json.dumps(payload,ensure_ascii=False)+"\n\n"
                last=result["status"]
            for item in store.query_events(query_id,seq):
                seq=item["seq"]
                yield "event: node\ndata: "+json.dumps(item,ensure_ascii=False)+"\n\n"
            if result["status"] in {"done","failed"}:
                return
            await asyncio.sleep(1)
        yield 'event: error\ndata: {"error":"Event stream timed out; poll query status"}\n\n'
    return StreamingResponse(events(),media_type="text/event-stream")


@app.get("/projects")
def projects():
    return store.projects()


@app.post("/projects/{project_id}/runs",status_code=201)
def add_run(project_id:str,run:RunInput):
    try:
        store.add_run(project_id,run)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    except Exception as exc:
        raise HTTPException(409,str(exc)) from exc
    if config.neo4j_password:
        try:
            graph=GraphStore(config)
            graph.project_run(project_id,run)
            graph.close()
        except Exception:
            pass  # SQLite is authoritative; graph can be rebuilt.
    return {"project_id":project_id,"run_id":run.run_id,"simulated":True}


@app.get("/projects/{project_id}/lineage")
def lineage(project_id:str):
    return {"project_id":project_id,"simulated":True,"runs":store.lineage(project_id)}


@app.post("/graph/query")
def graph_query(query:GraphQuery):
    if query.kind=="project_lineage":
        return {"kind":query.kind,"rows":store.lineage(query.project_id),"simulated":True}
    if not config.neo4j_password:
        raise HTTPException(503,"Neo4j is not configured")
    graph=GraphStore(config)
    try:
        if query.kind=="method_papers":
            rows=graph.method_papers(query.term)
        elif query.kind=="shared_datasets":
            rows=graph.shared_datasets(query.term,query.second_term)
        else:
            rows=graph.method_parameters(query.term)
        return {"kind":query.kind,"rows":rows}
    finally:
        graph.close()
