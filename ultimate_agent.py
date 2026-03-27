import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import operator
import time
import xml.etree.ElementTree as ET
import markdown
from io import BytesIO
from docx import Document
from htmldocx import HtmlToDocx
from typing import TypedDict, List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from inmoose.edgepy import DGEList, glmFit, glmLRT
from patsy import dmatrix
# --- NEW RAG IMPORTS ---
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==========================================
# PAGE CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="Agentic Oncology Orchestrator", layout="wide")

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Lab Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Lab Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop() # Stops the rest of the app from loading until password is correct!

st.title("🧬 Agentic Precision Oncology Pipeline")
st.markdown("Powered by LangGraph, PyDESeq2, OncoKB, and PubMed")

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    oncokb_key = st.secrets["ONCOKB_API_KEY"]
except KeyError:
    st.error("⚠️ Secrets not found! Please ensure you have a .streamlit/secrets.toml file with your API keys.")
    st.stop()

# --- INITIALIZE SESSION STATE (MEMORY) ---
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 1. GRAPH STATE & SCHEMAS
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    significant_genes: List[Dict[str, Any]]
    plan: List[str]
    gathered_evidence: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str
    custom_knowledge: str # <-- NEW: Slot for the RAG data

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
    search_query = f"{gene}[Gene] AND {tumor_type} AND targeted therapy"
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {"db": "pubmed", "term": search_query, "retmode": "json", "retmax": 3}
    
    try:
        res = requests.get(search_url, params=search_params)
        if res.status_code != 200:
            return {"status": f"PubMed Search Error: {res.status_code}"}
            
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return {"status": "No experimental literature found."}
            
        time.sleep(0.5) 
        
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
        
        fetch_res = requests.get(fetch_url, params=fetch_params)
        if fetch_res.status_code != 200:
            return {"status": f"PubMed Fetch Error: {fetch_res.status_code}"}
            
        papers = []
        root = ET.fromstring(fetch_res.content)
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else "Unknown"
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else "No Title"
            
            abstract_text = ""
            abstract_nodes = article.findall('.//AbstractText')
            if abstract_nodes:
                abstract_text = " ".join([node.text for node in abstract_nodes if node.text])
            else:
                abstract_text = "No abstract available."
                
            papers.append({
                "PMID": pmid, 
                "Title": title,
                "Abstract": abstract_text[:1000]
            })
            
        time.sleep(0.5)
        return {"status": "Success", "papers": papers}
        
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def search_clinical_trials(gene, tumor_type):
    url = "https://clinicaltrials.gov/api/v2/studies"
    query = f"{gene} AND {tumor_type}"
    params = {"query.cond": query, "filter.overallStatus": "RECRUITING", "pageSize": 3}
    
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json()
            studies = data.get("studies", [])
            if not studies:
                return {"status": "No recruiting trials found."}
                
            trials = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                design = protocol.get("designModule", {}) 
                
                nct_id = ident.get("nctId", "Unknown NCT")
                title = ident.get("briefTitle", "No Title")
                phase = ", ".join(design.get("phases", ["Phase Unknown"])) 
                
                trials.append({"NCT_ID": nct_id, "Title": title, "Phase": phase})
                
            time.sleep(0.5)
            return {"status": "Success", "trials": trials}
            
        return {"status": f"ClinicalTrials Error: {res.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def process_pdf_for_rag(pdf_file):
    """Reads a PDF, splits it into chunks, and builds a FAISS vector database."""
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            raw_text += extracted
            
    # CRITICAL: Prevent the database from crashing if the PDF is just images!
    if not raw_text.strip():
        return None 
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    return vectorstore

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def planner_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0, api_key=openai_key)
    structured_llm = llm.with_structured_output(Plan)
    
    sys_msg = """You are an expert Clinical Bioinformatics Planner. 
    Analyze the user prompt and genes. Output a step-by-step plan to gather data.
    Available Tools: 
    1. 'OncoKB' (FDA drugs)
    2. 'PubMed' (Experimental research)
    3. 'ClinicalTrials' (Actively recruiting trials)"""
    
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
        source_tag = gene_info.get("source", "Unknown Source")
        
        report = {"gene": hugo, "alteration": alt, "source": source_tag, "evidence": {}}
        
        if "oncokb" in plan_text:
            report["evidence"]["OncoKB"] = get_onco_data(hugo, alt, tumor_type)
        if "pubmed" in plan_text:
            report["evidence"]["PubMed"] = search_pubmed(hugo, tumor_type)
        if "clinicaltrials" in plan_text or "trials" in plan_text:
            print(f"   -> Fetching Clinical Trials for {hugo}...")
            report["evidence"]["ClinicalTrials"] = search_clinical_trials(hugo, tumor_type)
            
        new_evidence.append(report)
        
    return {"gathered_evidence": new_evidence}

def writer_node(state: AgentState):
    print("✍️ [NODE: Writer] Synthesizing the final clinical report...")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.2, api_key=openai_key)
    
    sys_msg = """You are an expert Clinical Oncology Medical Writer.
    Write a beautiful, multi-paragraph scientific report answering the user's prompt.
    
    CRITICAL CLINICAL TRIAGE RULES:
    1. Standard of Care (On-Label): List only drugs with Level 1 or Level 2 evidence. These are FDA-approved for the user's specific cancer type.
    2. Repurposing Opportunities (Off-Label): List only drugs with Level 3 (3A, 3B) or Level 4 evidence. These are approved for different cancers but show biomarker matches.
    
    CRITICAL GUARDRAILS: 
    1. ONLY use the data in the Gathered Evidence. Do NOT invent drugs or trials.
    2. For PubMed literature, read the Abstracts. Write a 1-2 sentence clinical summary of what the study actually found. Cite the PMID.
    3. Make all ClinicalTrials NCT IDs clickable markdown links.
    4. Important Context: Because this data originates from RNA sequencing (Overexpression), add a brief, single-sentence disclaimer at the top of the report stating: "*Note: Targeted therapies listed below typically require DNA confirmation of the specific mutation (e.g., V600E, L858R) associated with the overexpressed gene.*"
    5. Lab Protocols: If 'Custom Lab Protocols' are provided in the context, you MUST incorporate those specific internal rules, dosing guidelines, or protocol notes into the report.
    
    REQUIRED REPORT STRUCTURE:
    You MUST format your report using this EXACT Markdown structure for EVERY gene sequentially. Do not omit the PMIDs:
    
    ## [Gene Name] ([Alteration])
    
    ### 💊 OncoKB Therapeutics
    - **Standard of Care (On-Label):**
      - [Drug Name]: [Clinical Context] (PMIDs: [List PMIDs])
    - **Repurposing Opportunities (Off-Label):**
      - [Drug Name]: [Clinical Context] (PMIDs: [List PMIDs])
    
    ### 🔬 Experimental Literature
    - **[Study Topic/Focus]:** [1-2 sentence summary of abstract] (PMID: [Number])
    
    ### 🏥 Actively Recruiting Trials
    - **[[NCT ID]](https://clinicaltrials.gov/study/[NCT ID]):** [Phase] - [Trial Title]
    
    Do not deviate from this structure."""
    
    user_context = f"User Prompt: {state.get('user_prompt')}\nGathered Evidence: {json.dumps(state.get('gathered_evidence'))}\nCustom Lab Protocols: {state.get('custom_knowledge', 'None provided.')}"
    
    response = llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_context)
    ])
    
    print("✅ Final report successfully written.")
    return {"final_report": response.content}

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
# STREAMLIT FRONTEND & UI (VERSION 2.0)
# ==========================================
# Initialize session state variables so they survive button clicks
if "volcano_fig" not in st.session_state:
    st.session_state.volcano_fig = None
if "ai_targets" not in st.session_state:
    st.session_state.ai_targets = []

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Data Upload")
    counts_file = st.file_uploader("Upload RNA Counts (CSV)", type=["csv"])
    metadata_file = st.file_uploader("Upload Metadata (CSV)", type=["csv"])
    
    st.markdown("---")
    st.subheader("2. Statistical Cutoffs")
    # --- The Engine Selector and Form ---
    with st.form("stats_form"):
        de_engine = st.selectbox("Differential Expression Engine", ["PyDESeq2", "EdgePy"])
        pval_thresh = st.number_input("P-Value Cutoff", min_value=0.0001, max_value=0.1000, value=0.0500, step=0.0100, format="%.4f")
        log2fc_thresh = st.slider("Log2FC Threshold (Absolute)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        top_n_genes = st.slider("Max Targets for AI Report", min_value=1, max_value=15, value=3)
        
        update_plot_btn = st.form_submit_button("📊 Generate Volcano Plot")

    st.markdown("---")
    st.subheader("3. Clinical Context")
    cancer_type = st.text_input("Cancer Type (e.g., Melanoma, NSCLC)", value="Melanoma")
    
    # --- NEW RAG UI ---
    st.markdown("---")
    st.subheader("4. Custom Knowledge (Optional)")
    uploaded_pdf = st.file_uploader("Upload Lab Protocols/Guidelines (PDF)", type=["pdf"])
    
    st.markdown("---")
    run_button = st.button("🚀 Run AI Clinical Triage", use_container_width=True, type="primary")

with col2:
    st.subheader("Interactive Volcano Plot")
    
    # Only run the heavy math if files are uploaded AND the update button was clicked
    if counts_file and metadata_file and update_plot_btn:
        counts_df = pd.read_csv(counts_file, index_col=0)
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        
        with st.spinner(f"Calculating Differential Expression using {de_engine}..."):
            if de_engine == "PyDESeq2":
                # --- The original PyDESeq2 logic ---
                dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors="condition")
                dds.deseq2()
                stat_res = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"])
                stat_res.summary()
                results_df = stat_res.results_df
                
            elif de_engine == "EdgePy":
                # 1. Build the Design Matrix
                design = dmatrix("~condition", data=metadata_df)
                
                # 2. Initialize the EdgePy DGEList (Digital Gene Expression)
                dge_list = DGEList(counts=counts_df, samples=metadata_df, group_col="condition", genes=counts_df.index)
                
                # 3. Fit the Generalized Linear Model (GLM)
                fit = glmFit(dge_list, design=design)
                
                # 4. Run the Likelihood Ratio Test (LRT) for the 'condition' variable
                lrt = glmLRT(fit)
                
                # 5. Extract and format the results to match our PyDESeq2 shape
                # InMoose outputs pandas dataframes just like PyDESeq2!
                res = lrt.table
                results_df = pd.DataFrame(index=res.index)
                results_df['log2FoldChange'] = res['logFC']
                results_df['padj'] = res['FDR'] # EdgeR uses FDR instead of padj
            
        plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
        plot_df['-log10(padj)'] = -np.log10(plot_df['padj'] + 1e-300)
        
        conditions = [
            (plot_df['padj'] < pval_thresh) & (plot_df['log2FoldChange'] > log2fc_thresh), 
            (plot_df['padj'] < pval_thresh) & (plot_df['log2FoldChange'] < -log2fc_thresh)
        ]
        plot_df['Significance'] = np.select(conditions, ['Upregulated', 'Downregulated'], default='Not Significant')
        
        st.session_state.ai_targets = plot_df[plot_df['Significance'] == 'Upregulated'].sort_values(by='padj').head(top_n_genes).index.tolist()
        plot_df.loc[st.session_state.ai_targets, 'Significance'] = 'AI Selected Target'

        fig = px.scatter(
            plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', 
            color_discrete_map={
                'AI Selected Target': '#FFD700', 'Upregulated': '#EF553B', 
                'Downregulated': '#636EFA', 'Not Significant': '#4A4A4A' # NEW: Dark grey for better contrast
            },
            hover_name=plot_df.index
        )
        # --- NEW: Changed lines to white ---
        fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash", line_color="white")
        fig.add_vline(x=log2fc_thresh, line_dash="dash", line_color="white")
        fig.add_vline(x=-log2fc_thresh, line_dash="dash", line_color="white")
        fig.update_layout(height=500)
        
        st.session_state.volcano_fig = fig # Save plot to memory
        
    # Always display the plot if it exists in memory, even if they clicked a different button!
    if st.session_state.volcano_fig:
        st.plotly_chart(st.session_state.volcano_fig, use_container_width=True)
        
        if len(st.session_state.ai_targets) > 0:
            formatted_genes = ", ".join([f"`{gene}`" for gene in st.session_state.ai_targets])
            st.success(f"✅ **{len(st.session_state.ai_targets)} Targets identified:** {formatted_genes}")
        else:
            st.warning("⚠️ **No targets selected.** Adjust your statistical cutoffs and update the plot.")
    elif not counts_file or not metadata_file:
        st.info("👈 Upload data and click 'Generate Volcano Plot' to begin.")

# ==========================================
# EXECUTE THE AI GRAPH
# ==========================================
if run_button and counts_file and metadata_file and len(st.session_state.ai_targets) > 0:
    st.markdown("---")
    st.subheader("🤖 AI Clinical Report")
    
    # --- NEW: RAG PDF PROCESSING (BULLETPROOF VERSION) ---
    rag_context = ""
    if uploaded_pdf is not None:
        try:
            with st.spinner("📚 Reading uploaded Lab Protocol into Vector Database..."):
                vectorstore = process_pdf_for_rag(uploaded_pdf)
                
                if vectorstore is None:
                    st.warning("⚠️ Could not read text from this PDF (it might be a scanned image). Proceeding without custom knowledge.")
                else:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    query = f"Protocols, guidelines, and context for {cancer_type} or genes: {', '.join(st.session_state.ai_targets)}"
                    docs = retriever.invoke(query)
                    rag_context = "\n\n".join([d.page_content for d in docs])
                    st.success("✅ Custom Knowledge Base loaded and queried!")
                    
        except Exception as e:
            st.warning(f"⚠️ PDF Database Error: {str(e)}. Proceeding using only public data.")
    
    with st.spinner("Orchestrating AI Agents (Fetching OncoKB & PubMed)..."):
        structured_genes = []
        for gene in st.session_state.ai_targets:
            structured_genes.append({
                "hugo": gene,
                "alteration": "Overexpression", 
                "tumor_type": cancer_type,
                "source": "Volcanic Selection"
            })
            
        initial_state = {
            "user_prompt": f"Find targeted therapies for {cancer_type} patients with overexpression in {', '.join(st.session_state.ai_targets)}",
            "significant_genes": structured_genes,
            "plan": [],
            "gathered_evidence": [],
            "final_report": "",
            "custom_knowledge": rag_context # <-- NEW: Pass the PDF text to the AI!
        }
        
        final_state = orchestrator.invoke(initial_state)
        
        st.session_state.run_complete = True
        st.session_state.final_report = final_state["final_report"]
        st.session_state.plan = final_state["plan"]
        
# ==========================================
# 5. RENDER RESULTS & CHATBOT (From Memory)
# ==========================================
if st.session_state.run_complete:
    st.markdown("---")
    st.subheader("📈 Gene Expression Volcano Plot")
    st.plotly_chart(st.session_state.volcano_fig, use_container_width=True)
    
    with st.expander("🔍 View the AI's Strategic Plan"):
        for step in st.session_state.plan:
            st.write(f"- {step}")
            
    st.markdown("### 📄 Final Synthesized Clinical Report")
    st.info("This report was autonomously written by the Medical Writer LLM based solely on validated tool data.")
    st.markdown(st.session_state.final_report)
    
    # --- EXPORT MENU (HTML & DOCX) ---
    st.markdown("### 💾 Export Options")
    
    html_content = markdown.markdown(st.session_state.final_report, extensions=['tables'])
    
    styled_html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 40px auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                ul {{ margin-bottom: 20px; }}
                li {{ margin-bottom: 8px; }}
            </style>
        </head>
        <body>
            <h1>Clinical AI Orchestrator Report</h1>
            <p><strong>Disease Target:</strong> {cancer_type}</p>
            <hr>
            {html_content}
        </body>
    </html>
    """
    
    doc = Document()
    doc.add_heading(f'Clinical AI Orchestrator Report - {cancer_type}', level=1)
    
    parser = HtmlToDocx()
    parser.add_html_to_document(html_content, doc)
    
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0) 
    
    col_down1, col_down2 = st.columns(2)
    
    with col_down1:
        st.download_button(
            label="🌐 Download as HTML (Browser/PDF)",
            data=styled_html,
            file_name=f"{cancer_type}_Clinical_Report.html",
            mime="text/html",
            use_container_width=True
        )
        
    with col_down2:
        st.download_button(
            label="📄 Download as Word Document (.docx)",
            data=doc_buffer,
            file_name=f"{cancer_type}_Clinical_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )

    # --- INTERACTIVE CHATBOT ---
    st.markdown("---")
    st.subheader("💬 Discuss the Findings")
    st.write("Ask follow-up questions about the clinical trials, specific drugs, or resistance mechanisms mentioned above.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("E.g., What is the mechanism of action for CL-387785?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_llm = ChatOpenAI(model="gpt-5.2", temperature=0.2, api_key=openai_key)
                
                chat_sys_msg = f"You are a helpful oncology assistant. Answer the user's questions strictly based on the following report:\n\n{st.session_state.final_report}"
                
                messages = [SystemMessage(content=chat_sys_msg)]
                for m in st.session_state.messages:
                    if m["role"] == "user": messages.append(HumanMessage(content=m["content"]))
                    else: messages.append(AIMessage(content=m["content"]))
                    
                response = chat_llm.invoke(messages)
                st.markdown(response.content)
                
        st.session_state.messages.append({"role": "assistant", "content": response.content})