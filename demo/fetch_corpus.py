"""Download up to 30 open-access research PDFs and record provenance.

Run explicitly; this script is never invoked by the API. It verifies PDF headers and
records failures rather than replacing missing papers with fabricated documents.
"""
import hashlib
import json
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parent / "corpus"
ROOT.mkdir(exist_ok=True)
QUERIES=["seismic tomography velocity model", "ground penetrating radar geophysics", "seismic inversion machine learning"]
session=requests.Session()
session.headers["User-Agent"]="LabKnowledgeAgent/0.1 (research demo; contact via repository)"
records=[]
seen=set()
for phrase in QUERIES:
    response=session.get("https://api.openalex.org/works",params={"search":phrase,"filter":"is_oa:true,type:article","per-page":25},timeout=30)
    response.raise_for_status()
    for work in response.json().get("results",[]):
        if len(records)>=30:
            break
        doi=work.get("doi") or work["id"]
        if doi in seen:
            continue
        seen.add(doi)
        locations=[work.get("best_oa_location") or {}]+work.get("locations",[])
        urls=[loc.get("pdf_url") for loc in locations if loc and loc.get("pdf_url")]
        for url in dict.fromkeys(urls):
            try:
                pdf=session.get(url,timeout=60,allow_redirects=True)
                pdf.raise_for_status()
                if not pdf.content.startswith(b"%PDF") or len(pdf.content)>100*1024*1024:
                    continue
                digest=hashlib.sha256(pdf.content).hexdigest()
                path=ROOT / f"{digest[:16]}.pdf"
                path.write_bytes(pdf.content)
                records.append({"title":work.get("title"),"doi":doi,"source_url":url,
                                "sha256":digest,"file":path.name,"openalex_id":work["id"]})
                print(len(records),work.get("title"))
                break
            except (requests.RequestException,OSError):
                continue
    if len(records)>=30:
        break
(ROOT/"manifest.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Downloaded {len(records)} verified PDF files to {ROOT}")
