from typing import Any, Literal
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    id: str
    document_id: str
    version_id: str
    page_number: int
    block_id: str | None = None
    bbox: list[float] | None = None
    text: str = ""
    channel: str = ""
    score: float = 0.0


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["supported", "contradicted", "insufficient"] = "insufficient"
    reason: str = ""


class Answer(BaseModel):
    text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    trace_id: str = ""
    degraded: list[str] = Field(default_factory=list)


class RunInput(BaseModel):
    run_id: str
    input_version: str
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: Literal["planned", "running", "completed", "failed"]
    timestamp: str
    artifact: str
    parent_run_id: str | None = None


class QueryInput(BaseModel):
    question: str = Field(min_length=2)
    document_ids: list[str] = Field(default_factory=list)
    project_id: str | None = None


class GraphQuery(BaseModel):
    kind: Literal["method_papers", "shared_datasets", "method_parameters", "project_lineage"]
    term: str = ""
    second_term: str = ""
    project_id: str = ""


class Entity(BaseModel):
    name: str
    kind: Literal["Paper", "Method", "Dataset", "Area", "Experiment", "Parameter", "Metric", "Artifact"]
    aliases: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    source: str
    target: str
    kind: Literal["PROPOSES", "USES_METHOD", "USES_DATASET", "USES_PARAMETER", "PRODUCES", "DERIVED_FROM", "HAS_METRIC"]
    evidence_id: str
    context: str = ""


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
