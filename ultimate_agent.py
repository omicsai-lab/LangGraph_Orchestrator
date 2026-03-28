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
    pathway_data: Dict[str, Any] # <-- NEW: Holds the Enrichr KEGG pathways
    final_report: str
    custom_knowledge: str 
    analysis_mode: str

class Plan(BaseModel):
    steps: List[str] = Field(description="Step-by-step plan of tools to execute.")

# ==========================================
# 2. THE TOOLS (Python Functions)
# ==========================================
def get_gene_info(hugo_symbol):
    """Fetches biological context, gene type, and aliases."""
    url = f"https://mygene.info/v3/query?q=symbol:{hugo_symbol}&fields=name,summary,type_of_gene,alias&species=human"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("hits"):
                hit = data["hits"][0]
                # Format aliases nicely whether it's a string or a list
                aliases = hit.get("alias", [])
                if isinstance(aliases, list):
                    aliases = ", ".join(aliases)
                return {
                    "name": hit.get("name", "Unknown"),
                    "type": hit.get("type_of_gene", "Unknown"),
                    "summary": hit.get("summary", "No summary available."),
                    "aliases": aliases
                }
        return {"status": "Gene info not found."}
    except Exception as e:
        return {"status": f"API Error: {str(e)}"}

def get_enriched_pathways(gene_list):
    """Uses the Enrichr API to find overlapping biological pathways for a list of genes."""
    if not gene_list:
        return "No genes provided for pathway analysis."
        
    enrichr_add_url = 'https://maayanlab.cloud/Enrichr/addList'
    payload = {
        'list': (None, '\n'.join(gene_list)),
        'description': (None, 'OncoApp_Gene_List')
    }
    
    try:
        # 1. Upload the list to Enrichr
        res_add = requests.post(enrichr_add_url, files=payload)
        if not res_add.ok: return "Enrichr API upload failed."
        user_list_id = res_add.json().get('userListId')
        
        # 2. Query the KEGG Pathway Database
        enrichr_query_url = f'https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType=KEGG_2021_Human'
        res_query = requests.get(enrichr_query_url)
        if not res_query.ok: return "Enrichr KEGG query failed."
        
        results = res_query.json().get('KEGG_2021_Human', [])
        
        # 3. Extract the top 3 most significant pathways
        top_pathways = []
        for r in results[:3]:
            top_pathways.append({
                "pathway": r[1],
                "p_value": r[2],
                "overlapping_genes": r[5]
            })
        return {"status": "Success", "pathways": top_pathways}
    except Exception as e:
        return {"status": f"Enrichr Request failed: {str(e)}"}

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

def search_pubmed(gene, tumor_type, mode="Clinical Triage", aliases=""):
    # Clean up the aliases into a search string if they exist
    alias_query = ""
    if aliases and aliases != "Unknown":
        # Take up to the first 2 aliases to prevent massive URL queries
        alias_list = [a.strip() for a in aliases.split(',')][:2]
        if alias_list:
            alias_query = " OR " + " OR ".join([f"{a}[Title/Abstract]" for a in alias_list])

    if "Discovery" in mode:
        search_query = f"({gene}[Gene]{alias_query}) AND {tumor_type} AND (immunotherapy OR biomarker OR novel target)"
    else:
        search_query = f"({gene}[Gene]{alias_query}) AND {tumor_type} AND targeted therapy"
        
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
    
    # NEW: Run Pathway Analysis for the entire group of targets
    print("   -> Running KEGG Pathway Analysis via Enrichr...")
    gene_symbols = [g.get("hugo") for g in genes]
    pathway_results = get_enriched_pathways(gene_symbols)
    
    for gene_info in genes:
        hugo = gene_info.get("hugo")
        alt = gene_info.get("alteration")
        tumor_type = gene_info.get("tumor_type")
        source_tag = gene_info.get("source", "Unknown Source")
        
        # NEW: Automatically fetch the biological definition first!
        print(f"   -> Fetching biological context for {hugo}...")
        gene_context = get_gene_info(hugo)
        
        report = {"gene": hugo, "alteration": alt, "source": source_tag, "biology": gene_context, "evidence": {}}
        
        if "oncokb" in plan_text:
            report["evidence"]["OncoKB"] = get_onco_data(hugo, alt, tumor_type)
        if "pubmed" in plan_text:
            # Pass the mode AND the aliases into the PubMed search
            report["evidence"]["PubMed"] = search_pubmed(
                hugo, 
                tumor_type, 
                mode=state.get("analysis_mode", "Clinical Triage"),
                aliases=gene_context.get("aliases", "")
            )
        if "clinicaltrials" in plan_text or "trials" in plan_text:
            print(f"   -> Fetching Clinical Trials for {hugo}...")
            report["evidence"]["ClinicalTrials"] = search_clinical_trials(hugo, tumor_type)
            
        new_evidence.append(report)
        
    return {"gathered_evidence": new_evidence, "pathway_data": pathway_results}

def writer_node(state: AgentState):
    print("✍️ [NODE: Writer] Synthesizing the final clinical report...")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.2, api_key=openai_key)
    
    analysis_mode = state.get("analysis_mode", "Clinical Triage")
    
    if "Discovery" in analysis_mode:
        sys_msg = """You are an expert Systems Biologist and Bioinformatics AI.
        Write a beautiful, pathway-centric scientific report answering the user's prompt. 
        
        CRITICAL GUARDRAILS:
        1. TONE AND STYLE: NEVER break the fourth wall. Do NOT say "per your guardrails", "in the provided evidence", or "your PubMed pull". Write confidently as if you are authoring a published review article in a high-impact oncology journal.
        2. BIOLOGICAL TRIAGE: Explicitly dismiss pseudogenes and ncRNAs as non-coding artifacts.
        3. ACRONYM COLLISIONS: Be highly aware of literature false-positives. If PubMed returns papers where the gene symbol is used as an acronym for a drug (e.g., CEL = Celastrol) or a biological process (e.g., LPO = Lipid Peroxidation), YOU MUST EXPLICITLY CALL THIS OUT as a literature mismatch. Do not treat the paper as evidence for the gene.
        4. SYSTEMS APPROACH: Do NOT list genes one by one. Group them by their pathway and discuss them as a network.
        
        REQUIRED REPORT STRUCTURE:
        ## 🕸️ Systems Biology & Pathway Dysregulation
        [Write a multi-paragraph synthesis of the KEGG pathway data. How do these networks (and their overlapping genes) interact to drive the tumor microenvironment, metabolic reprogramming, or immune evasion?]
        
        ## 🔬 Targetable Hubs & Experimental Literature
        [Synthesize the PubMed literature conceptually. Discuss the valid genes as a group. If papers suffered from Acronym Collisions, note that the literature is currently lacking for the specific genes due to nomenclature overlap.]
        
        ## 🏥 Translational Outlook
        [Summarize any relevant trials, or state that these novel network targets currently lack specific recruiting trials.]
        """
    else:
        sys_msg = """You are an expert Clinical Oncology Medical Writer.
        Write a beautiful, multi-paragraph scientific report answering the user's prompt.
        
        CRITICAL CLINICAL TRIAGE RULES:
        1. Standard of Care: Level 1 or 2 evidence.
        2. Repurposing: Level 3 or 4 evidence.
        3. Dismiss pseudogenes/ncRNAs using the biology context.
        
        REQUIRED REPORT STRUCTURE:
        First, provide a brief summary of the overarching biological pathways driving this tumor:
        ### 🕸️ Pathway & Network Dysregulation
        - [Write a 2-3 sentence summary based on the KEGG pathway_data provided. Mention the overlapping genes.]
        
        Then, for EVERY gene sequentially, use this exact structure:
        ## [Gene Name] ([Alteration])
        
        ### 💊 OncoKB Therapeutics
        - **Standard of Care (On-Label):** [Drug Name] (PMIDs: [List])
        - **Repurposing Opportunities (Off-Label):** [Drug Name] (PMIDs: [List])
        
        ### 🔬 Experimental Literature
        - **[Study Topic]:** [Summary] (PMID: [Number])
        
        ### 🏥 Actively Recruiting Trials
        - **[[NCT ID]](https://clinicaltrials.gov/study/[NCT ID]):** [Phase] - [Trial Title]
        
        Do not deviate from this structure."""
    
    user_context = f"User Prompt: {state.get('user_prompt')}\nPathway Data: {json.dumps(state.get('pathway_data', {}))}\nGathered Evidence: {json.dumps(state.get('gathered_evidence'))}\nCustom Lab Protocols: {state.get('custom_knowledge', 'None provided.')}"
    
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
    st.subheader("Optional: DNA Mutational Profile")
    dna_file = st.file_uploader("Upload DNA Variants (CSV with 'Gene' and 'Alteration' columns)", type=["csv"])
    
    # --- NEW: Dynamic Covariate Selection ---
    condition_col = "condition" # Fallbacks
    batch_col = "None"
    
    if metadata_file is not None:
        # Peek at the metadata columns without locking up memory
        temp_meta = pd.read_csv(metadata_file, nrows=0) 
        meta_cols = temp_meta.columns.tolist()
        
        # CRITICAL FIX: Rewind the file pointer back to the beginning!
        metadata_file.seek(0)
        
        st.markdown("---")
        st.subheader("2. Experimental Design")
        col_a, col_b = st.columns(2)
        with col_a:
            condition_col = st.selectbox("Primary Contrast (e.g., Tumor vs Normal)", meta_cols, index=meta_cols.index("condition") if "condition" in meta_cols else 0)
        with col_b:
            batch_col = st.selectbox("Batch Covariate (Optional)", ["None"] + meta_cols)
            
    st.markdown("---")
    st.subheader("3. Statistical Cutoffs")
    # --- The Engine Selector and Form ---
    with st.form("stats_form"):
        de_engine = st.selectbox("Differential Expression Engine", ["PyDESeq2", "EdgePy"])
        pval_thresh = st.number_input("P-Value Cutoff", min_value=0.0001, max_value=0.1000, value=0.0500, step=0.0100, format="%.4f")
        log2fc_thresh = st.slider("Log2FC Threshold (Absolute)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        
        update_plot_btn = st.form_submit_button("📊 Generate Volcano Plot")

    st.markdown("---")
    st.subheader("3. Clinical Context & AI Triage")
    cancer_type = st.text_input("Cancer Type (e.g., Melanoma, NSCLC)", value="Melanoma")
    analysis_mode = st.radio("Analysis Mode", ["Clinical Triage (Known Targets)", "Biomarker Discovery (Novel Targets)"])
    top_n_genes = st.slider("Max Targets for AI Report", min_value=1, max_value=15, value=3)
    
    # --- NEW RAG UI ---
    st.markdown("---")
    st.subheader("5. Custom Knowledge (Optional)")
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
            # Determine the design formula strings based on user selection
            if batch_col != "None" and batch_col != condition_col:
                design_factors = [batch_col, condition_col]
                edge_formula = f"~{batch_col} + {condition_col}"
            else:
                design_factors = condition_col
                edge_formula = f"~{condition_col}"

            # Auto-detect the contrast levels from the primary column
            unique_levels = metadata_df[condition_col].dropna().unique()
            level_1 = unique_levels[0]
            level_2 = unique_levels[1] if len(unique_levels) > 1 else unique_levels[0]

            if de_engine == "PyDESeq2":
                # --- Updated PyDESeq2 logic with covariates ---
                dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors=design_factors)
                dds.deseq2()
                stat_res = DeseqStats(dds, contrast=[condition_col, level_1, level_2])
                stat_res.summary()
                results_df = stat_res.results_df
                
            elif de_engine == "EdgePy":
                # 1. Build the Design Matrix with optional batch effect
                design = dmatrix(edge_formula, data=metadata_df)
                
                # 2. Initialize the EdgePy DGEList
                dge_list = DGEList(counts=counts_df, samples=metadata_df, group_col=condition_col, genes=counts_df.index)
                
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
        
        # Save all upregulated genes to memory for the Actionability Filter
        st.session_state.upregulated_df = plot_df[plot_df['Significance'] == 'Upregulated'].sort_values(by='padj')

        # Generate a clean map of the tumor
        fig = px.scatter(
            plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', 
            color_discrete_map={
                'Upregulated': '#EF553B', 
                'Downregulated': '#636EFA', 'Not Significant': '#4A4A4A' 
            },
            hover_name=plot_df.index,
            render_mode='webgl' 
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
# ==========================================
# EXECUTE THE AI GRAPH
# ==========================================
if run_button and counts_file and metadata_file:
    st.markdown("---")
    st.subheader("🤖 AI Clinical Report")
    
    # --- NEW: THE ACTIONABILITY FILTER ---
    ACTIONABLE_GENES = ["BRAF", "EGFR", "KRAS", "PIK3CA", "ERBB2", "ALK", "ROS1", "MET", "RET", "NTRK1", "NTRK2", "NTRK3", "BRCA1", "BRCA2", "KIT", "PDGFRA", "FGFR1", "FGFR2", "FGFR3", "IDH1", "IDH2", "CDK4", "CDK6", "PTEN", "MTOR", "CTNNB1", "TP53"]
    
    up_df = st.session_state.get("upregulated_df", pd.DataFrame())
    if up_df.empty:
        st.error("⚠️ No upregulated genes found. Please lower your P-Value or Log2FC thresholds in the Volcano Plot first.")
        st.stop()
        
    if "Discovery" in analysis_mode:
        # NOVEL: Filter OUT the famous genes
        novel_df = up_df[~up_df.index.isin(ACTIONABLE_GENES)]
        st.session_state.ai_targets = novel_df.head(top_n_genes).index.tolist()
    else:
        # CLINICAL: ONLY look at famous genes
        clinical_df = up_df[up_df.index.isin(ACTIONABLE_GENES)]
        st.session_state.ai_targets = clinical_df.head(top_n_genes).index.tolist()
        
    if not st.session_state.ai_targets:
        st.warning(f"⚠️ No targets found for {analysis_mode} mode. Try adjusting your statistical cutoffs.")
        st.stop()
        
    st.success(f"🎯 **Target Selection Complete:** {', '.join(st.session_state.ai_targets)}")
    
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
        dna_gene_names = []
        
        # 1. Parse Optional DNA Mutations (Highest Priority for FDA Drugs)
        if dna_file is not None:
            try:
                dna_df = pd.read_csv(dna_file)
                if 'Gene' in dna_df.columns and 'Alteration' in dna_df.columns:
                    for _, row in dna_df.iterrows():
                        gene_name = str(row['Gene']).strip()
                        dna_gene_names.append(gene_name)
                        structured_genes.append({
                            "hugo": gene_name,
                            "alteration": str(row['Alteration']).strip(),
                            "tumor_type": cancer_type,
                            "source": "DNA Mutation (Level 1/2 Priority)"
                        })
                dna_file.seek(0)
            except Exception as e:
                st.warning(f"⚠️ Could not parse DNA file: {str(e)}")
        
        # 2. Add RNA Overexpression Targets
        for gene in st.session_state.ai_targets:
            structured_genes.append({
                "hugo": gene,
                "alteration": "Overexpression", 
                "tumor_type": cancer_type,
                "source": "RNA Volcanic Selection"
            })
            
        # 3. Smart Prompt Generation (Handling both DNA and RNA)
        if "Discovery" in analysis_mode:
            prompt_text = f"Analyze the following overexpressed genes ({', '.join(st.session_state.ai_targets)}) in {cancer_type} as potential novel biomarkers or immunotherapeutic targets."
            if dna_gene_names:
                prompt_text += f" Also contextualize the presence of these specific DNA mutations: {', '.join(dna_gene_names)}."
        else:
            prompt_text = f"Find established targeted therapies for {cancer_type} patients."
            if dna_gene_names:
                prompt_text += f" CRITICAL: Prioritize finding OncoKB Level 1/2 FDA-approved therapies for the following DNA mutations: {', '.join(dna_gene_names)}."
            if st.session_state.ai_targets:
                prompt_text += f" Secondary: Evaluate the following overexpressed targets: {', '.join(st.session_state.ai_targets)}."

        initial_state = {
            "user_prompt": prompt_text,
            "significant_genes": structured_genes,
            "plan": [],
            "gathered_evidence": [],
            "pathway_data": {},
            "final_report": "",
            "custom_knowledge": rag_context, # <-- NEW: Pass the PDF text to the AI!
            "analysis_mode": analysis_mode
        }
        
        final_state = orchestrator.invoke(initial_state)
        
        st.session_state.run_complete = True
        st.session_state.final_report = final_state["final_report"]
        st.session_state.plan = final_state["plan"]
        st.session_state.pathway_data = final_state.get("pathway_data", {}) # <-- NEW: Extract pathway data
        
# ==========================================
# 5. RENDER RESULTS & CHATBOT (From Memory)
# ==========================================
if st.session_state.run_complete:
    st.markdown("---")
    st.subheader("📈 Gene Expression Volcano Plot")
    st.plotly_chart(st.session_state.volcano_fig, use_container_width=True, key="bottom_volcano_plot")
    
    with st.expander("🔍 View the AI's Strategic Plan"):
        for step in st.session_state.plan:
            st.write(f"- {step}")
            
    # --- NEW: PATHWAY VISUALIZATION ---
    pathway_info = st.session_state.get("pathway_data", {})
    if isinstance(pathway_info, dict) and pathway_info.get("status") == "Success":
        st.markdown("### 🕸️ Enriched Biological Pathways (KEGG)")
        pathways = pathway_info.get("pathways", [])
        
        if pathways:
            # Convert to DataFrame for Plotly
            pw_df = pd.DataFrame(pathways)
            # -log10 transform the p-value for better visualization
            pw_df['Significance Score (-log10 p-value)'] = -np.log10(pw_df['p_value'] + 1e-10)
            
            # Draw horizontal bar chart
            # Draw horizontal bar chart
            pw_fig = px.bar(
                pw_df, 
                x='Significance Score (-log10 p-value)', 
                y='pathway', 
                orientation='h',
                title="Top Associated KEGG Pathways",
                text='overlapping_genes',
                color='Significance Score (-log10 p-value)',
                color_continuous_scale='Sunsetdark' # A warmer, less clunky dark mode palette
            )
            # Clean up the layout, text position, and hover decimals
            pw_fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=300, margin=dict(l=20, r=20, t=40, b=20))
            pw_fig.update_traces(
                textposition='inside', 
                textfont=dict(color='white'),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<br>Genes: %{text}<extra></extra>"
            )
            st.plotly_chart(pw_fig, use_container_width=True)
        else:
            st.info("No statistically significant pathways found for these targets.")

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