import argparse
import json
from pathlib import Path

from .config import settings
from .ingest import Ingestor
from .models import QueryInput, RunInput
from .store import Store
from .workflow import Agent


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    add=sub.add_parser("ingest"); add.add_argument("pdf",type=Path)
    ask=sub.add_parser("ask"); ask.add_argument("question")
    sub.add_parser("seed-demo")
    args=parser.parse_args()
    config=settings(); store=Store(config.db_path)
    if args.command=="ingest":
        ingestor=Ingestor(config,store)
        doc_id=ingestor.register(args.pdf.name,args.pdf.read_bytes())
        print(json.dumps(ingestor.process(doc_id),ensure_ascii=False))
    elif args.command=="ask":
        print(Agent(config,store).ask(QueryInput(question=args.question)).model_dump_json(indent=2))
    else:
        demo=json.loads((Path(__file__).resolve().parents[2]/"demo"/"projects.json").read_text(encoding="utf-8"))
        for project_id,runs in demo.items():
            for run in runs:
                try:
                    store.add_run(project_id,RunInput.model_validate(run))
                except Exception as exc:
                    if "UNIQUE" not in str(exc):
                        raise
        print("Seeded",len(demo),"simulated projects")
