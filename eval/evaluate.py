"""Evaluate a frozen, manually labeled JSONL benchmark without inventing scores."""
import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def evaluate(path:Path, api:str, variant:str):
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Benchmark is empty")
    if any(not r.get("question") or "gold_pages" not in r for r in rows):
        raise ValueError("Every row needs question and gold_pages")
    results=[]
    for row in rows:
        started=time.perf_counter()
        query=requests.post(api+"/queries",json={"question":row["question"],"document_ids":row.get("document_ids",[]),"project_id":row.get("project_id")},timeout=20)
        query.raise_for_status()
        query_id=query.json()["query_id"]
        for _ in range(120):
            run=requests.get(api+f"/queries/{query_id}",timeout=20).json()
            if run["status"] in {"done","failed"}:
                break
            time.sleep(1)
        else:
            run={"status":"failed","error":"timeout"}
        elapsed=time.perf_counter()-started
        if run["status"]!="done":
            results.append({"id":row.get("id"),"error":run.get("error"),"latency_s":elapsed})
            continue
        answer=run["result"]
        ranked=[(e["document_id"],e["page_number"]) for e in answer["evidence"]]
        gold={tuple(x) for x in row["gold_pages"]}
        results.append({"id":row.get("id"),"latency_s":elapsed,
                        "recall_at_5":bool(gold.intersection(ranked[:5])) if gold else None,
                        "recall_at_10":bool(gold.intersection(ranked[:10])) if gold else None,
                        "claims":answer["claims"],"unresolved":answer["unresolved"],
                        "answer":answer["text"],"degraded":answer["degraded"]})
    eligible=[r for r in results if "recall_at_5" in r and r["recall_at_5"] is not None]
    times=[r["latency_s"] for r in results]
    times_sorted=sorted(times)
    report={"variant":variant,"sample_count":len(rows),"completed":sum("error" not in r for r in results),
            "recall_at_5":sum(r["recall_at_5"] for r in eligible)/len(eligible) if eligible else None,
            "recall_at_10":sum(r["recall_at_10"] for r in eligible)/len(eligible) if eligible else None,
            "p50_latency_s":statistics.median(times),"p95_latency_s":times_sorted[min(len(times)-1,int(len(times)*0.95))],
            "note":"Claim correctness and citation support require independent human labels; this script does not infer them from model output.",
            "results":results}
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("benchmark",type=Path)
    parser.add_argument("--api",default="http://localhost:8000")
    parser.add_argument("--variant",default="full")
    parser.add_argument("--output",type=Path,default=Path("eval/report.json"))
    args=parser.parse_args()
    result=evaluate(args.benchmark,args.api,args.variant)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="results"},ensure_ascii=False,indent=2))
