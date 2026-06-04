import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import operator
import time
from typing import TypedDict, List, Dict, Any, Annotated
from pydantic import BaseModel, Field

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==========================================
# PAGE CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="Agentic Oncology Orchestrator", layout="wide")
st.title("🧬 Agentic Precision Oncology Pipeline")
st.markdown("Powered by LangGraph, PyDESeq2, OncoKB, and PubMed")

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    oncokb_key = st.secrets["ONCOKB_API_KEY"]
except KeyError:
    st.error("⚠️ Secrets not found! Please ensure you have a .streamlit/secrets.toml file with your API keys.")
    st.stop()

# ==========================================
# 1. GRAPH STATE & SCHEMAS
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    significant_genes: List[Dict[str, Any]]
    plan: List[str]
    gathered_evidence: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str

class Plan(BaseModel):
    steps: List[str] = Field(description="Step-by-step plan of tools to execute.")

# ==========================================
# 2. THE TOOLS (Python Functions)
# ==========================================
def get_onco_data(hugo, alteration, tumor_type):
    url = "https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange"
    params = {"hugoSymbol": hugo, "alteration": alteration, "tumorType": tumor_type}
    headers = {"accept": "application/json", "Authorization": f"Bearer {oncokb_key}"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            treatments = data.get('treatments', [])
            if not treatments: return {"status": "No drug entries found."}
            
            results = []
            for treatment in treatments:
                drugs = [d.get('drugName', '') for d in treatment.get('drugs', [])]
                results.append({
                    "drugName": ", ".join(drugs), 
                    "levelOfEvidence": treatment.get('level', 'Unknown'),
                    "pmids": treatment.get('pmids', [])
                })
            return {"status": "Success", "drugs": results}
        return {"status": f"OncoKB Error: {response.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def search_pubmed(gene, tumor_type):
    """Hits the free PubMed API (Two-step: Search for IDs, then fetch Titles)"""
    search_query = f"{gene}[Gene] AND {tumor_type} AND targeted therapy"
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {"db": "pubmed", "term": search_query, "retmode": "json", "retmax": 3}
    
    try:
        # Step 1: Get the list of PMIDs
        res = requests.get(search_url, params=search_params)
        if res.status_code != 200:
            return {"status": f"PubMed Search Error: {res.status_code}"}
            
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return {"status": "No experimental literature found."}
            
        # --- THE FIX: Pause for half a second to respect PubMed's rate limit ---
        time.sleep(0.5) 
        
        # Step 2: Fetch the Titles for those specific PMIDs
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}
        
        sum_res = requests.get(summary_url, params=summary_params)
        if sum_res.status_code != 200:
            return {"status": f"PubMed Summary Error: {sum_res.status_code}"}
            
        sum_data = sum_res.json().get("result", {})
        
        papers = []
        for pmid in id_list:
            title = sum_data.get(pmid, {}).get("title", "Title not available")
            papers.append({"PMID": pmid, "Title": title})
            
        # Pause one more time before the loop moves to the next gene
        time.sleep(0.5)
            
        return {"status": "Success", "papers": papers}
        
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def planner_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
    structured_llm = llm.with_structured_output(Plan)
    
    sys_msg = """You are an expert Clinical Bioinformatics Planner. 
    Analyze the user prompt and genes. Output a step-by-step plan to gather data.
    Available Tools: 'OncoKB' (FDA drugs), 'PubMed' (Experimental research)."""
    
    context = f"User Prompt: {state.get('user_prompt')}\nGenes: {state.get('significant_genes')}"
    response = structured_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=context)])
    
    return {"plan": response.steps}

def executor_node(state: AgentState):
    plan_text = " ".join(state.get("plan", [])).lower()
    genes = state.get("significant_genes", [])
    new_evidence = []
    
    for gene_info in genes:
        hugo = gene_info.get("hugo")
        alt = gene_info.get("alteration")
        tumor_type = gene_info.get("tumor_type")
        
        report = {"gene": hugo, "alteration": alt, "evidence": {}}
        
        # Guardrails enforced!
        if "oncokb" in plan_text:
            report["evidence"]["OncoKB"] = get_onco_data(hugo, alt, tumor_type)
        if "pubmed" in plan_text:
            report["evidence"]["PubMed"] = search_pubmed(hugo, tumor_type)
            
        new_evidence.append(report)
        
    return {"gathered_evidence": new_evidence}

def writer_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_key)
    
    sys_msg = """You are a Clinical Oncology Medical Writer.
    Write a beautiful, multi-paragraph scientific report answering the user's prompt.
    CRITICAL GUARDRAILS: 
    1. ONLY use the data in the Gathered Evidence. 
    2. Do NOT invent drugs or clinical trials.
    3. For PubMed literature, you will be provided with PMIDs and Titles. You MUST NOT invent summaries, abstracts, or outcomes for these papers. Simply list the Title and the PMID as a citation.
    Format nicely with Markdown headers, bold text, and bullet points."""
    
    context = f"User Prompt: {state.get('user_prompt')}\nEvidence: {json.dumps(state.get('gathered_evidence'))}"
    response = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=context)])
    
    return {"final_report": response.content}

# Compile the Graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "writer")
workflow.add_edge("writer", END)

orchestrator = workflow.compile()

# ==========================================
# 4. STREAMLIT UI & MAIN EXECUTION
# ==========================================
st.subheader("1. Configure Analysis")
col1, col2 = st.columns(2)
with col1:
    counts_file = st.file_uploader("Upload RNA-seq Counts (CSV)", type=["csv"])
    metadata_file = st.file_uploader("Upload Metadata (CSV)", type=["csv"])
with col2:
    disease_interest = st.text_input("Cancer Type", value="Melanoma")
    user_prompt = st.text_area(
        "📝 Instructions for the AI Orchestrator", 
        value="Identify potential therapeutics for the top targets. If a target is un-drugged in OncoKB, highlight experimental PubMed literature."
    )

if st.button("Run Agentic Pipeline", type="primary"):
    if counts_file and metadata_file:
        counts_df = pd.read_csv(counts_file, index_col=0)
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        
        with st.status("⚙️ Processing Omics Data...", expanded=True) as status:
            st.write("Running PyDESeq2 Differential Expression...")
            dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors="condition")
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"])
            stat_res.summary()
            results_df = stat_res.results_df
            status.update(label="PyDESeq2 Complete!", state="complete", expanded=False)
            
        # --- VOLCANO PLOT ---
        st.markdown("---")
        st.subheader("📈 Gene Expression Volcano Plot")
        plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
        plot_df['-log10(padj)'] = -np.log10(plot_df['padj'] + 1e-300)
        conditions = [(plot_df['padj'] < 0.05) & (plot_df['log2FoldChange'] > 2), (plot_df['padj'] < 0.05) & (plot_df['log2FoldChange'] < -2)]
        choices = ['Upregulated (Target)', 'Downregulated']
        plot_df['Significance'] = np.select(conditions, choices, default='Not Significant')
        fig = px.scatter(plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', color_discrete_map={'Upregulated (Target)': 'red', 'Downregulated': 'blue', 'Not Significant': 'lightgrey'}, hover_name=plot_df.index)
        st.plotly_chart(fig, use_container_width=True)

        # Prepare payload for the AI
        sig_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] > 2)].sort_values(by='padj')
        agent_payload = [{"hugo": str(g), "alteration": "V600E" if g == "BRAF" else "L858R" if g == "EGFR" else "Amplification", "tumor_type": disease_interest} for g in sig_genes.index]
        
        # --- LANGGRAPH EXECUTION ---
        st.markdown("---")
        st.subheader("🧠 LangGraph AI Orchestration")
        with st.spinner("The AI is planning, executing tools, and writing the report..."):
            initial_state = {
                "user_prompt": user_prompt,
                "significant_genes": agent_payload,
                "plan": [],
                "gathered_evidence": [],
                "final_report": ""
            }
            
            # This single line runs the entire board game we just built!
            final_state = orchestrator.invoke(initial_state)
            
        # Display the AI's internal thoughts (The Plan)
        with st.expander("🔍 View the AI's Strategic Plan"):
            for step in final_state["plan"]:
                st.write(f"- {step}")
                
        # Display the finalized Medical Report
        st.markdown("### 📄 Final Synthesized Clinical Report")
        st.info("This report was autonomously written by the Medical Writer LLM based solely on validated tool data.")
        st.markdown(final_state["final_report"])
            
    else:
        st.warning("Please upload both the Counts and Metadata CSV files to begin.")