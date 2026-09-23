import re

from neo4j import GraphDatabase

from .config import Settings
from .models import Extraction, RunInput


def normalized(name: str) -> str:
    return re.sub(r"\s+", " ", name.casefold()).strip()


class GraphStore:
    LABELS = {"Paper","Method","Dataset","Area","Experiment","Parameter","Metric","Artifact"}
    RELS = {"PROPOSES","USES_METHOD","USES_DATASET","USES_PARAMETER","PRODUCES","DERIVED_FROM","HAS_METRIC"}

    def __init__(self, config: Settings):
        if not config.neo4j_password:
            raise RuntimeError("LAB_NEO4J_PASSWORD is required")
        self.driver = GraphDatabase.driver(config.neo4j_uri,auth=(config.neo4j_user,config.neo4j_password))

    def close(self):
        self.driver.close()

    def add_extraction(self, extraction: Extraction, document_id: str, page_number: int):
        names = {normalized(e.name):e for e in extraction.entities}
        with self.driver.session() as session:
            for e in extraction.entities:
                if e.kind not in self.LABELS:
                    continue
                # Labels are selected from a fixed enum, never inserted from user text.
                session.run(f"MERGE (n:{e.kind} {{key:$key}}) SET n.name=$name,n.aliases=$aliases",
                            key=f"{e.kind}:{normalized(e.name)}",name=e.name,aliases=e.aliases)
            for r in extraction.relations:
                source=names.get(normalized(r.source)); target=names.get(normalized(r.target))
                if not source or not target or r.kind not in self.RELS or not r.evidence_id.startswith(f"{document_id}:{page_number}:"):
                    continue
                session.run(f"MATCH (a:{source.kind} {{key:$a}}),(b:{target.kind} {{key:$b}}) MERGE (a)-[r:{r.kind} {{evidence_id:$evidence}}]->(b) SET r.document_id=$document,r.page_number=$page,r.context=$context",
                            a=f"{source.kind}:{normalized(source.name)}",b=f"{target.kind}:{normalized(target.name)}",
                            evidence=r.evidence_id,document=document_id,page=page_number,context=r.context)

    def related_pages(self, terms: list[str], limit: int = 10):
        if not terms:
            return []
        query="""MATCH (n)-[r]-(m) WHERE any(t IN $terms WHERE toLower(n.name) CONTAINS toLower(t))
                 AND r.document_id IS NOT NULL RETURN DISTINCT r.document_id AS document_id,r.page_number AS page_number,r.evidence_id AS evidence_id,r.context AS context LIMIT $limit"""
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query,terms=terms[:8],limit=limit)]

    def method_papers(self, term: str):
        with self.driver.session() as session:
            return [dict(r) for r in session.run("MATCH (p:Paper)-[e:PROPOSES]->(m:Method) WHERE toLower(m.name) CONTAINS toLower($term) RETURN p.name AS paper,m.name AS method,e.evidence_id AS evidence_id LIMIT 20",term=term)]

    def shared_datasets(self, method_a: str, method_b: str):
        query="""MATCH (a:Experiment)-[:USES_METHOD]->(m1:Method),(b:Experiment)-[:USES_METHOD]->(m2:Method),
                 (a)-[:USES_DATASET]->(d:Dataset)<-[:USES_DATASET]-(b)
                 WHERE toLower(m1.name) CONTAINS toLower($a) AND toLower(m2.name) CONTAINS toLower($b)
                 RETURN DISTINCT d.name AS dataset LIMIT 20"""
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query,a=method_a,b=method_b)]

    def method_parameters(self, term: str):
        query="""MATCH (e:Experiment)-[:USES_METHOD]->(m:Method),(e)-[r:USES_PARAMETER]->(p:Parameter)
                 WHERE toLower(m.name) CONTAINS toLower($term)
                 RETURN e.name AS experiment,m.name AS method,p.name AS parameter,r.context AS context,r.evidence_id AS evidence_id LIMIT 20"""
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query,term=term)]

    def project_run(self, project_id: str, run: RunInput):
        with self.driver.session() as session:
            session.run("MERGE (e:Experiment {key:$key}) SET e.name=$name,e.project_id=$project,e.status=$status,e.simulated=true",
                        key=f"run:{project_id}:{run.run_id}",name=run.run_id,project=project_id,status=run.status)
            session.run("MERGE (a:Artifact {key:$key}) SET a.name=$name,a.simulated=true",
                        key=f"artifact:{project_id}:{run.artifact}",name=run.artifact)
            session.run("MATCH (e:Experiment {key:$run}),(a:Artifact {key:$artifact}) MERGE (e)-[:PRODUCES]->(a)",
                        run=f"run:{project_id}:{run.run_id}",artifact=f"artifact:{project_id}:{run.artifact}")
            if run.parent_run_id:
                session.run("MATCH (e:Experiment {key:$run}),(p:Experiment {key:$parent}) MERGE (e)-[:DERIVED_FROM]->(p)",
                            run=f"run:{project_id}:{run.run_id}",parent=f"run:{project_id}:{run.parent_run_id}")
