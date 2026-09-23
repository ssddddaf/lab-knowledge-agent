import os
import time

import requests
import streamlit as st

BASE=os.getenv("LAB_API_URL","http://localhost:8000")
st.set_page_config(page_title="实验室知识 Agent",layout="wide")
st.title("实验室知识 Agent")
st.caption("论文证据与模拟项目记录；模拟项目不代表真实实验室运行")


def get(path):
    response=requests.get(BASE+path,timeout=20)
    response.raise_for_status()
    return response.json()


documents,qa,projects=st.tabs(["文档库","问答与证据","模拟项目追踪"])

with documents:
    uploaded=st.file_uploader("上传 PDF",type="pdf")
    if uploaded and st.button("开始入库"):
        response=requests.post(BASE+"/documents",files={"file":(uploaded.name,uploaded.getvalue(),"application/pdf")},timeout=30)
        response.raise_for_status()
        st.session_state["job_id"]=response.json()["job_id"]
    if st.session_state.get("job_id"):
        st.json(get("/jobs/"+st.session_state["job_id"]))
    try:
        st.dataframe(get("/documents"),use_container_width=True)
    except Exception as exc:
        st.error(f"API 未连接：{exc}")

with qa:
    question=st.text_area("问题",placeholder="比较两篇论文使用的数据集和速度模型；证据来自哪一页？")
    doc_filter=st.text_input("文档 ID 过滤（可选，逗号分隔）")
    project_filter=st.text_input("模拟项目 ID（可选）")
    if st.button("查询") and question.strip():
        payload={"question":question,"document_ids":[x.strip() for x in doc_filter.split(",") if x.strip()],"project_id":project_filter or None}
        response=requests.post(BASE+"/queries",json=payload,timeout=20)
        response.raise_for_status()
        st.session_state["query_id"]=response.json()["query_id"]
    if st.session_state.get("query_id"):
        result=get("/queries/"+st.session_state["query_id"])
        if result["status"] in {"queued","running"}:
            st.info("正在检索与校验…")
            time.sleep(1)
            st.rerun()
        elif result["status"]=="failed":
            st.error(result["error"])
        elif result["result"]:
            answer=result["result"]
            st.markdown(answer["text"])
            if result.get("events"):
                st.caption("执行节点："+" → ".join(e["node"] for e in result["events"]))
            if answer["degraded"]:
                st.warning("降级状态："+", ".join(answer["degraded"]))
            if answer["unresolved"]:
                st.info("未解决："+"；".join(answer["unresolved"]))
            with st.expander("断言校验与执行 ID"):
                st.write("执行 ID：",answer["trace_id"])
                st.json(answer["claims"])
            for e in answer["evidence"]:
                with st.expander(f'{e["document_id"][:12]} · 第 {e["page_number"]} 页 · {e["channel"]}'):
                    st.write(e["text"])
                    if e["page_number"]>0:
                        if e["bbox"] and e["block_id"]:
                            st.image(BASE+f'/documents/{e["document_id"]}/pages/{e["page_number"]}/regions/{e["block_id"]}',caption="定位到的证据区域")
                        st.image(BASE+f'/documents/{e["document_id"]}/pages/{e["page_number"]}',caption="来源页面；区域坐标见元数据" if e["bbox"] else "来源整页")
                        if e["bbox"]:
                            st.caption(f'区域坐标：{e["bbox"]}')

with projects:
    try:
        choices=get("/projects")
        project=st.selectbox("模拟项目",choices) if choices else None
        if project:
            lineage=get(f"/projects/{project}/lineage")
            st.dataframe(lineage["runs"],use_container_width=True)
            for run in lineage["runs"]:
                if run["parent_run_id"]:
                    st.write(f'{run["parent_run_id"]} → {run["run_id"]} → {run["artifact"]}')
        with st.form("add_run"):
            st.subheader("新增模拟处理记录")
            project_id=st.text_input("项目 ID",value=project or "")
            run_id=st.text_input("运行 ID")
            input_version=st.text_input("输入版本")
            method=st.text_input("方法")
            parameters=st.text_area("参数 JSON",value="{}")
            status=st.selectbox("状态",["planned","running","completed","failed"])
            timestamp=st.text_input("时间（ISO 8601）")
            artifact=st.text_input("产物")
            parent_run_id=st.text_input("父运行 ID（可选）")
            submitted=st.form_submit_button("保存模拟记录")
        if submitted:
            import json
            try:
                payload={"run_id":run_id,"input_version":input_version,"method":method,"parameters":json.loads(parameters),
                         "status":status,"timestamp":timestamp,"artifact":artifact,"parent_run_id":parent_run_id or None}
                response=requests.post(BASE+f"/projects/{project_id}/runs",json=payload,timeout=20)
                response.raise_for_status()
                st.success("模拟记录已保存")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    except Exception as exc:
        st.error(f"API 未连接：{exc}")
