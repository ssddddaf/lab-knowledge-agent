import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import RunInput


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, version_id TEXT NOT NULL, filename TEXT NOT NULL,
              path TEXT NOT NULL, status TEXT NOT NULL, error TEXT DEFAULT '',
              page_count INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS pages (
              document_id TEXT, page_number INTEGER, image_path TEXT NOT NULL,
              text TEXT DEFAULT '', PRIMARY KEY(document_id,page_number)
            );
            CREATE TABLE IF NOT EXISTS blocks (
              id TEXT PRIMARY KEY, document_id TEXT, page_number INTEGER,
              kind TEXT, text TEXT, bbox TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(id UNINDEXED,text);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, document_id TEXT, status TEXT, error TEXT DEFAULT '',
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS runs (
              project_id TEXT, run_id TEXT, input_version TEXT, method TEXT,
              parameters TEXT, status TEXT, timestamp TEXT, artifact TEXT,
              parent_run_id TEXT, simulated INTEGER NOT NULL DEFAULT 1,
              PRIMARY KEY(project_id,run_id)
            );
            CREATE TABLE IF NOT EXISTS queries (
              id TEXT PRIMARY KEY, status TEXT, result TEXT DEFAULT '', error TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS query_events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT, node TEXT,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def add_document(self, id: str, filename: str, path: str):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO documents(id,version_id,filename,path,status) VALUES(?,?,?,?,?)", (id,id,filename,path,"pending"))

    def document(self, id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (id,)).fetchone()
            return dict(row) if row else None

    def documents(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM documents ORDER BY created_at DESC")]

    def set_document_status(self, id: str, status: str, error: str = "", pages: int = 0):
        with self.connect() as db:
            db.execute("UPDATE documents SET status=?,error=?,page_count=? WHERE id=?", (status,error,pages,id))

    def add_page(self, document_id: str, number: int, image_path: str, text: str):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?,?)", (document_id,number,image_path,text))

    def page(self, document_id: str, number: int):
        with self.connect() as db:
            row = db.execute("SELECT * FROM pages WHERE document_id=? AND page_number=?", (document_id,number)).fetchone()
            return dict(row) if row else None

    def add_blocks(self, blocks: list[dict]):
        with self.connect() as db:
            for b in blocks:
                db.execute("INSERT OR REPLACE INTO blocks VALUES(?,?,?,?,?,?)", (b["id"],b["document_id"],b["page_number"],b["kind"],b["text"],json.dumps(b.get("bbox"))))
                db.execute("DELETE FROM blocks_fts WHERE id=?", (b["id"],))
                db.execute("INSERT INTO blocks_fts VALUES(?,?)", (b["id"],b["text"]))

    def blocks(self, document_ids: list[str] | None = None):
        with self.connect() as db:
            rows = db.execute("SELECT b.* FROM blocks b JOIN documents d ON d.id=b.document_id WHERE d.status='ready'").fetchall()
        result = [dict(r) for r in rows]
        return [r for r in result if r["document_id"] in document_ids] if document_ids else result

    def add_run(self, project_id: str, run: RunInput):
        with self.connect() as db:
            if run.parent_run_id and not db.execute("SELECT 1 FROM runs WHERE project_id=? AND run_id=?", (project_id,run.parent_run_id)).fetchone():
                raise ValueError("parent_run_id does not exist in this project")
            if run.parent_run_id == run.run_id:
                raise ValueError("run cannot derive from itself")
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,1)", (project_id,run.run_id,run.input_version,run.method,json.dumps(run.parameters,ensure_ascii=False),run.status,run.timestamp,run.artifact,run.parent_run_id))

    def lineage(self, project_id: str):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs WHERE project_id=? ORDER BY timestamp,run_id", (project_id,)).fetchall()
        return [{**dict(r),"parameters":json.loads(r["parameters"]),"simulated":True} for r in rows]

    def projects(self):
        with self.connect() as db:
            return [r[0] for r in db.execute("SELECT DISTINCT project_id FROM runs ORDER BY project_id")]

    def job(self, id: str):
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE id=?",(id,)).fetchone()
            return dict(row) if row else None

    def set_job(self, id: str, document_id: str, status: str, error: str = ""):
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,document_id,status,error) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,error=excluded.error,updated_at=CURRENT_TIMESTAMP", (id,document_id,status,error))

    def set_query(self, id: str, status: str, result: str = "", error: str = ""):
        with self.connect() as db:
            db.execute("INSERT INTO queries(id,status,result,error) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,result=excluded.result,error=excluded.error", (id,status,result,error))

    def query(self, id: str):
        with self.connect() as db:
            row=db.execute("SELECT * FROM queries WHERE id=?",(id,)).fetchone()
            return dict(row) if row else None

    def add_query_event(self, id: str, node: str):
        with self.connect() as db:
            db.execute("INSERT INTO query_events(query_id,node) VALUES(?,?)",(id,node))

    def query_events(self, id: str, after: int = 0):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM query_events WHERE query_id=? AND seq>? ORDER BY seq",(id,after))]
