import json
import re
import sqlite3
import uuid
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from .config import Settings
from .llm import LLM
from .models import Answer, Claim, Evidence, QueryInput
from .retrieval import Retriever
from .store import Store


def merge_evidence(old:list[dict], new:list[dict]) -> list[dict]:
    result={e["id"]:e for e in old}
    result.update({e["id"]:e for e in new})
    return list(result.values())


def merge_unique(old:list[str], new:list[str]) -> list[str]:
    return list(dict.fromkeys(old+new))


class QueryState(TypedDict, total=False):
    question: str
    document_ids: list[str]
    project_id: str | None
    tasks: list[str]
    evidence: Annotated[list[dict],merge_evidence]
    selected: list[dict]
    draft: str
    claims: list[dict]
    unresolved: list[str]
    retry_count: int
    degraded: Annotated[list[str],merge_unique]
    trace_id: str
    final: dict


class PlanResponse(BaseModel):
    subtasks: list[str] = Field(default_factory=list)


class DraftResponse(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    claims: list[Claim]
    missing_requirements: list[str] = Field(default_factory=list)


class Agent:
    def __init__(self, config: Settings, store: Store):
        self.config=config
        self.store=store
        self.llm=LLM(config)
        self.conn=sqlite3.connect(config.checkpoint_path,check_same_thread=False)
        self.graph=self._build().compile(checkpointer=SqliteSaver(self.conn))

    def _plan(self,state:QueryState):
        question=state["question"]
        if not self.llm.enabled:
            return {"tasks":[question]}
        try:
            response=self.llm.json(PlanResponse,"Break a question into independently searchable factual tasks, at most four. Keep a simple question as one task. Return original language.",question)
            return {"tasks":response.subtasks[:4] or [question]}
        except Exception:
            return {"tasks":[question]}

    @staticmethod
    def _fanout(state:QueryState):
        return [Send("retrieve",{"question":task,"document_ids":state.get("document_ids",[]),"project_id":state.get("project_id")}) for task in state.get("tasks",[])[:4]]

    def _retrieve(self,state:QueryState):
        retriever=Retriever(self.config,self.store)
        evidence=retriever.search(state["question"],state.get("document_ids"),state.get("project_id"),limit=6)
        return {"evidence":[e.model_dump() for e in evidence],"degraded":retriever.degraded}

    def _join(self,state:QueryState):
        items=[Evidence.model_validate(e) for e in state.get("evidence",[])]
        items=sorted(items,key=lambda x:x.score,reverse=True)
        # Keep some evidence from each source document when possible.
        selected=[]; per_doc={}
        for e in items:
            if per_doc.get(e.document_id,0)>=5:
                continue
            selected.append(e.model_dump()); per_doc[e.document_id]=per_doc.get(e.document_id,0)+1
            if len(selected)==12:
                break
        return {"selected":selected}

    def _draft(self,state:QueryState):
        evidence=[Evidence.model_validate(e) for e in state.get("selected",[])]
        if not evidence:
            return {"draft":"未找到足够证据回答该问题。","claims":[],"unresolved":[state["question"]]}
        context=[{"id":e.id,"document":e.document_id,"page":e.page_number,"text":e.text[:1800]} for e in evidence]
        if self.llm.enabled:
            try:
                images=[]
                for e in evidence:
                    if e.page_number>0:
                        page=self.store.page(e.document_id,e.page_number)
                        if page and e.channel=="visual":
                            from pathlib import Path
                            images.append(Path(page["image_path"]))
                answer=self.llm.json(DraftResponse,"Answer only using supplied evidence. Every factual claim must cite exact evidence IDs. Preserve units and experimental conditions. State unknowns explicitly.",
                    json.dumps({"question":state["question"],"evidence":context},ensure_ascii=False),images=images[:4])
                return {"draft":answer.answer,"claims":[c.model_dump() for c in answer.claims],"unresolved":[]}
            except Exception:
                pass
        snippets=[f'{e.text[:350]} [{e.id}]' for e in evidence[:4]]
        return {"draft":"检索到以下相关证据，请核对原文：\n"+"\n".join(snippets),"claims":[],"unresolved":["未配置或无法调用生成模型，未生成综合结论"]}

    def _verify(self,state:QueryState):
        claims=[Claim.model_validate(c) for c in state.get("claims",[])]
        if not claims:
            return {"unresolved":state.get("unresolved",[])}
        evidence={e["id"]:e for e in state.get("selected",[])}
        for claim in claims:
            cited=[evidence[x] for x in claim.evidence_ids if x in evidence]
            if not cited:
                claim.status="insufficient"; claim.reason="citation missing from retrieved evidence"
                continue
            numbers=set(re.findall(r"\d+(?:\.\d+)?%?",claim.text))
            source=" ".join(e["text"] for e in cited)
            if numbers and not numbers.issubset(set(re.findall(r"\d+(?:\.\d+)?%?",source))):
                claim.status="insufficient"; claim.reason="number absent from cited evidence"
            else:
                claim.status="supported"; claim.reason="citation present; numeric check passed"
        missing=[]
        if self.llm.enabled:
            try:
                checked=self.llm.json(VerifyResponse,"Independently assess whether each claim follows from its cited evidence, including method, dataset and units. Mark contradicted or insufficient when unsupported. Check every requested comparison dimension. Do not invent evidence.",
                    json.dumps({"question":state["question"],"answer":state.get("draft",""),"claims":[c.model_dump() for c in claims],"evidence":list(evidence.values())},ensure_ascii=False))
                by_text={c.text:c for c in checked.claims}
                for claim in claims:
                    other=by_text.get(claim.text)
                    if claim.status=="supported" and other and other.status!="supported":
                        claim.status=other.status; claim.reason=other.reason
                missing=checked.missing_requirements
            except Exception:
                missing=["语义校验服务不可用；仅完成引用与数字检查"]
        unresolved=[c.text+": "+c.reason for c in claims if c.status!="supported"]+missing
        return {"claims":[c.model_dump() for c in claims],"unresolved":unresolved}

    @staticmethod
    def _route(state:QueryState):
        if state.get("unresolved") and state.get("retry_count",0)<2 and state.get("claims"):
            return "retry"
        return "finish"

    def _retry(self,state:QueryState):
        count=state.get("retry_count",0)+1
        target=" ".join(state.get("unresolved",[])[:2])[:350]
        return {"retry_count":count,"question":state["question"],"tasks":state.get("tasks",[])+[target]}

    def _retry_retrieve(self,state:QueryState):
        retriever=Retriever(self.config,self.store)
        target=state.get("tasks",[])[-1]
        evidence=retriever.search(target,state.get("document_ids"),state.get("project_id"),limit=6)
        return {"evidence":[e.model_dump() for e in evidence],"degraded":retriever.degraded}

    def _finish(self,state:QueryState):
        answer=Answer(text=state.get("draft",""),claims=[Claim.model_validate(c) for c in state.get("claims",[])],
            evidence=[Evidence.model_validate(e) for e in state.get("selected",[])],
            unresolved=state.get("unresolved",[]),trace_id=state["trace_id"],
            degraded=state.get("degraded",[]) + (["model_unavailable"] if not self.llm.enabled else []))
        return {"final":answer.model_dump()}

    def _build(self):
        builder=StateGraph(QueryState)
        for name,func in [("plan",self._plan),("retrieve",self._retrieve),("join",self._join),
                          ("draft",self._draft),("verify",self._verify),("retry",self._retry),
                          ("retry_retrieve",self._retry_retrieve),("finish",self._finish)]:
            builder.add_node(name,func)
        builder.add_edge(START,"plan")
        builder.add_conditional_edges("plan",self._fanout,["retrieve"])
        builder.add_edge("retrieve","join")
        builder.add_edge("join","draft")
        builder.add_edge("draft","verify")
        builder.add_conditional_edges("verify",self._route,{"retry":"retry","finish":"finish"})
        builder.add_edge("retry","retry_retrieve")
        builder.add_edge("retry_retrieve","join")
        builder.add_edge("finish",END)
        return builder

    def ask(self, query:QueryInput, trace_id:str|None=None, on_node:Callable[[str],None]|None=None) -> Answer:
        trace_id=trace_id or str(uuid.uuid4())
        initial={"question":query.question,"document_ids":query.document_ids,"project_id":query.project_id,
                 "evidence":[],"claims":[],"retry_count":0,"unresolved":[],"degraded":[],"trace_id":trace_id}
        config={"configurable":{"thread_id":trace_id},"recursion_limit":24,"max_concurrency":3}
        if on_node:
            final=None
            for update in self.graph.stream(initial,config=config,stream_mode="updates"):
                for node,value in update.items():
                    on_node(node)
                    if node=="finish":
                        final=value["final"]
            if final is None:
                raise RuntimeError("Graph completed without an answer")
            return Answer.model_validate(final)
        output=self.graph.invoke(initial,config=config)
        return Answer.model_validate(output["final"])
