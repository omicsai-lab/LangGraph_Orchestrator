# ==========================================
# 🛡️ ONCOLOGY AGENT: PRISTINE IMPORT BLOCK
# ==========================================

# --- 1. CORE UI & SYSTEM ---
import streamlit as st
import streamlit.components.v1 as components
import os
import time
import json
import re
import operator  
import tempfile
import io
from io import BytesIO
import requests
import markdown
import copy
from typing import TypedDict, List, Dict, Any, Annotated
from pydantic import BaseModel, Field

# --- 2. DATA SCIENCE & MATH ---
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
import plotly.express as px
from tme_core import TMECore

# --- 3. GENOMICS & BIOINFORMATICS ---
import anndata as ad
import scanpy as sc  
import gseapy as gp
from gseapy.plot import gseaplot
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# --- 4. VISUALIZATION ---
import plotly.express as px
import matplotlib.pyplot as plt
from pyvis.network import Network
import py3Dmol
from stmol import showmol

# --- 5. REPORT GENERATION ---
from docx import Document
from htmldocx import HtmlToDocx
from PyPDF2 import PdfReader

# --- 6. AI BRAIN & RAG (LANGCHAIN/LANGGRAPH) ---
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document as LCDocument
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

# --- 7. LOCAL MODULES ---
from stats_engine import run_differential_stats 

if 'ai_targets' not in st.session_state: st.session_state.ai_targets = []
if 'volcano_fig' not in st.session_state: st.session_state.volcano_fig = None
if 'clinical_context' not in st.session_state: st.session_state.clinical_context = {}

# ==========================================
# PAGE CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="Agentic Oncology Orchestrator", layout="wide")

# --- PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
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
    st.stop() 

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
if "gathering_complete" not in st.session_state:
    st.session_state.gathering_complete = False
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ==========================================
# 1. GRAPH STATE & SCHEMAS
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    significant_genes: List[Dict[str, Any]]
    plan: List[str]
    gathered_evidence: Annotated[List[Dict[str, Any]], operator.add]
    pathway_data: Dict[str, Any] 
    final_report: str
    custom_knowledge: str 
    analysis_mode: str
    biomarker_intent: str 
    max_deep_dive: int             
    fast_triage_data: List[Dict[str, Any]]
    selection_logic: str
    discarded_evidence: List[Dict[str, Any]] 
    ai_filtered_evidence: List[Dict[str, Any]]
    expert_consensus: str
    tme_deconvolution: Dict[str, Any]
    therapeutic_modality: str 

class Plan(BaseModel):
    steps: List[str] = Field(description="Step-by-step plan of tools to execute.")

class PaperScore(BaseModel):
    score: int = Field(description="Relevance score from 1 to 10")
    reason: str = Field(description="Short 3-15 word reason (e.g., 'Acronym Collision', 'Strong evidence', 'Wrong Disease')")

class KnowledgePath(BaseModel):
    gene: str = Field(description="Hugo symbol")
    path: str = Field(description="Strict DR.KNOWS format. E.g., '[HPA: Macrophage] -> EXCLUDE (Artifact)'")
    status: str = Field(description="'INCLUDE' or 'EXCLUDE'")

# ==========================================
# 2. THE TOOLS (Python Functions)
# ==========================================

@st.cache_data(ttl="1d", show_spinner=False)
def get_gene_info(hugo_symbol):
    url = f"https://mygene.info/v3/query?q=symbol:{hugo_symbol}&fields=name,summary,type_of_gene,alias&species=human"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("hits"):
                hit = data["hits"][0]
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

@st.cache_data(ttl="1d", show_spinner=False)
def fetch_normal_tissue_profile(hugo_symbol):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0, api_key=openai_key)
    sys_msg = """You are a Genotype-Tissue Expression (GTEx) and Human Protein Atlas database proxy. 
    Output a strict, 1-sentence summary of where this gene is predominantly expressed in normal, healthy human tissue. 
    Be highly specific (e.g., 'Predominantly expressed in the exocrine pancreas and lactating mammary glands'). 
    If it is ubiquitously expressed across all tissues, explicitly state 'Ubiquitously expressed'."""
    try:
        res = llm.invoke([
            SystemMessage(content=sys_msg), 
            HumanMessage(content=f"Gene: {hugo_symbol}")
        ])
        return res.content
    except Exception as e:
        return "GTEx proxy unavailable."

@st.cache_resource(ttl="1d", show_spinner=False)
def run_gsea_analysis(full_df):
    df = full_df.dropna(subset=['log2FoldChange', 'padj']).copy()
    df['rank_metric'] = -np.log10(df['padj'] + 1e-300) * np.sign(df['log2FoldChange'])
    df = df.sort_values('rank_metric', ascending=False)
    rnk = df[['rank_metric']]
    
    try:
        try:
            pre_res = gp.prerank(
                rnk=rnk, 
                gene_sets='KEGG_2021_Human',
                threads=4, min_size=5, max_size=1000,
                permutation_num=100, outdir=None, seed=42
            )
        except Exception as thread_e:
            print(f"⚠️ Multiprocessing warning. Falling back to single thread...")
            pre_res = gp.prerank(
                rnk=rnk, gene_sets='KEGG_2021_Human',
                threads=1, min_size=5, max_size=1000,
                permutation_num=100, outdir=None, seed=42
            )
        
        res_df = pre_res.res2d
        sig_pw = res_df[res_df['FDR q-val'] < 0.05].head(10)
        
        if sig_pw.empty:
            return {"status": "No statistically significant pathways found by GSEA.", "pathways": []}
            
        top_pathways = []
        for idx, row in sig_pw.iterrows():
            lead_genes = row['Lead_genes'].split(';')
            top_pathways.append({
                "pathway": row['Term'],
                "p_value": row['NOM p-val'],
                "nes": row.get('NES', 0),
                "overlapping_genes": lead_genes
            })
            
        return {"status": "Success", "pathways": top_pathways, "gsea_obj": pre_res}
    except Exception as e:
        return {"status": f"GSEA failed: {str(e)}"}

@st.cache_data(ttl="1d", show_spinner=False)
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

@st.cache_data(ttl="1d", show_spinner=False)
def search_pubmed(gene, tumor_type, mode="Clinical Triage", intent="Therapeutic", aliases="", interactors=None):
    if interactors is None: interactors = []
    
    alias_query = ""
    if aliases and aliases != "Unknown":
        alias_list = [a.strip() for a in aliases.split(',') if len(a.strip()) > 3][:2]
        if alias_list:
            alias_query = " OR " + " OR ".join([f"{a}[TIAB]" for a in alias_list])

    if "Discovery" in mode and interactors:
        network_nodes = [gene] + interactors
        network_query_str = " OR ".join([f"{n}[TIAB]" for n in network_nodes])
        broad_query = f"({network_query_str}{alias_query}) AND {tumor_type}[TIAB]"
        prov_step_1 = f"**Phase 1 (Broad Network Pull):** Expanded query to include STRING interactors: `[{broad_query}]`."
    else:
        broad_query = f"({gene}[TIAB]{alias_query}) AND {tumor_type}[TIAB]"
        prov_step_1 = f"**Phase 1 (Broad Target Pull):** PubMed query `[{broad_query}]`."
    
    if "Diagnostic" in intent:
        semantic_query = f"Diagnostic biomarker, liquid biopsy, ELISA blood test, early detection, risk stratification, and prognostic survival outcomes in {tumor_type}."
    elif "Discovery" in mode:
        semantic_query = f"Novel biomarkers, signaling pathways, lipid metabolism, immunotherapy targets, and resistance mechanisms in {tumor_type}."
    else:
        semantic_query = f"FDA approved targeted therapy, survival outcomes, and clinical trial results for {tumor_type}."

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {"db": "pubmed", "term": broad_query, "retmode": "json", "retmax": 40}
    
    try:
        res = requests.get(search_url, params=search_params)
        if res.status_code != 200: return {"status": f"Search Error: {res.status_code}"}
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list: return {"status": "No experimental literature found."}
            
        time.sleep(0.5) 
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
        fetch_res = requests.get(fetch_url, params=fetch_params)
        
        papers = []
        import xml.etree.ElementTree as ET
        root = ET.fromstring(fetch_res.content)
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else "Unknown"
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else "No Title"
            abstract_nodes = article.findall('.//AbstractText')
            abstract_text = " ".join([node.text for node in abstract_nodes if node.text]) if abstract_nodes else ""
            if abstract_text: 
                papers.append({"PMID": pmid, "Title": title, "Abstract": abstract_text[:1500]})
                
        if not papers: 
            return {"status": "No abstracts passed the initial filter.", "papers": [], "provenance": [prov_step_1 + " Yielded 0 relevant candidates."]}

        st.markdown(f"      -> Embedding {len(papers)} abstracts into FAISS for {gene}...")
        docs = [LCDocument(page_content=f"Title: {p['Title']}\nAbstract: {p['Abstract']}", metadata=p) for p in papers]
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        relevant_docs = retriever.invoke(semantic_query)
        top_papers = [{"PMID": d.metadata["PMID"], "Title": d.metadata["Title"], "Abstract": d.metadata["Abstract"]} for d in relevant_docs]
        
        provenance = [
            prov_step_1 + f" Yielded {len(id_list)} candidates.",
            f"**Phase 2 (Semantic Sorting):** Embedded {len(papers)} valid abstracts into FAISS Vector DB.",
            f"**Phase 3 (Concept Retrieval):** Extracted top 20 papers using diagnostic-aware query: *'{semantic_query}'*.",
            f"**Phase 4 (Expert Review):** Top semantic matches passed to the AI Scorer for strict triage."
        ]
        return {"status": "Success", "papers": top_papers, "provenance": provenance}
        
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

@st.cache_data(ttl="1d", show_spinner=False)
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

@st.cache_data(show_spinner=False)
def check_biomarker_detectability(gene_symbol: str) -> str:
    url = f"https://www.proteinatlas.org/api/search_download.php?search={gene_symbol}&format=json&columns=g,pc,sec,pt&compress=no"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            try:
                data = res.json()
            except ValueError:
                return "HPA API returned an invalid response."
                
            for entry in data:
                if entry.get("Gene", "").upper() == gene_symbol.upper():
                    classes = entry.get("Protein class", [])
                    classes_str = ", ".join(classes) if isinstance(classes, list) else str(classes)
                    
                    is_secreted = "Secreted" in classes_str or "Plasma" in classes_str or "Predicted secreted" in classes_str
                    is_membrane = "Membrane" in classes_str or "Predicted membrane" in classes_str
                    
                    if is_secreted:
                        return f"DETECTABILITY: High. {gene_symbol} is a secreted protein, making it a prime candidate for ELISA blood tests or liquid biopsies."
                    elif is_membrane:
                        return f"DETECTABILITY: Moderate. {gene_symbol} is a membrane protein. It may be detectable via flow cytometry/CTCs."
                    else:
                        return f"DETECTABILITY: Low. {gene_symbol} is intracellular. Clinical detection requires an invasive tissue biopsy."
            return f"No detectability/secretome data found for {gene_symbol}."
        else:
            return f"Secretome API Error: {res.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

@st.cache_data(ttl="1d", show_spinner=False)
def get_protein_interactions(hugo_symbol):
    url = f"https://string-db.org/api/json/network?identifiers={hugo_symbol}&species=9606&limit=3"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if not data: return {"status": "No interactions found."}
            
            interactors = []
            for edge in data:
                neighbor = edge.get("preferredName_B") if edge.get("preferredName_A") == hugo_symbol else edge.get("preferredName_A")
                if neighbor and neighbor not in interactors:
                    interactors.append(neighbor)
            
            return {"status": "Success", "interacting_proteins": interactors[:3]}
        return {"status": f"STRING API Error: {res.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

@st.cache_data(ttl="1d", show_spinner=False)
def fetch_target_tractability(hugo_symbol):
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    query = """
    query targetSearch($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"]) {
        hits {
          object {
            ... on Target {
              id
              approvedSymbol
              tractability { label modality value }
              depMapEssentiality { screens { depmapId diseaseFromSource } }
            }
          }
        }
      }
    }
    """
    variables = {"queryString": hugo_symbol}
    try:
        res = requests.post(url, json={"query": query, "variables": variables})
        if res.status_code == 200:
            data = res.json()
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            for hit in hits:
                obj = hit.get("object", {})
                if obj and obj.get("approvedSymbol") == hugo_symbol:
                    tractability = obj.get("tractability") or []
                    is_druggable = False
                    modalities = []
                    for t in tractability:
                        if t.get("value") == True:
                            is_druggable = True
                            modalities.append(f"{t.get('modality')} ({t.get('label')})")
                    
                    essentiality_data = obj.get("depMapEssentiality") or []
                    is_essential = False
                    essential_screens = 0
                    if isinstance(essentiality_data, list) and len(essentiality_data) > 0:
                        screens = essentiality_data[0].get("screens", [])
                        essential_screens = len(screens)
                        is_essential = essential_screens > 0
                    
                    status_msg = "Success" if is_druggable or is_essential else "Target exists but has NO tractability or DepMap data."
                    return {
                        "status": status_msg,
                        "is_druggable": is_druggable,
                        "tractability_buckets": modalities[:5],
                        "is_depmap_essential": is_essential,
                        "essential_cell_lines": essential_screens
                    }
            return {"status": f"Target '{hugo_symbol}' not found in OpenTargets Database."}
        return {"status": f"API Error: {res.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def process_pdf_for_rag(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = "".join([page.extract_text() or "" for page in reader.pages])
    if not raw_text.strip(): return None 
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    return FAISS.from_texts(chunks, embeddings)

@st.cache_data(ttl="1d", show_spinner=False)
def get_uniprot_id(hugo_symbol):
    url = f"https://mygene.info/v3/query?q=symbol:{hugo_symbol}&fields=uniprot&species=human"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("hits"):
                uniprot = data["hits"][0].get("uniprot", {})
                if "Swiss-Prot" in uniprot: return uniprot["Swiss-Prot"]
                elif "TrEMBL" in uniprot: return uniprot["TrEMBL"][0] if isinstance(uniprot["TrEMBL"], list) else uniprot["TrEMBL"]
        return None
    except: return None

def get_tcga_population_frequency(gene_symbol: str, cancer_type: str) -> str:
    cancer_map = {
        "breast": "brca_tcga_pan_can_atlas_2018", "melanoma": "skcm_tcga_pan_can_atlas_2018",
        "lung": "luad_tcga_pan_can_atlas_2018", "squamous lung": "lusc_tcga_pan_can_atlas_2018",
        "colon": "coadread_tcga_pan_can_atlas_2018", "prostate": "prad_tcga_pan_can_atlas_2018",
        "brain": "gbm_tcga_pan_can_atlas_2018", "pancreas": "paad_tcga_pan_can_atlas_2018"
    }
    
    study_id = next((val for key, val in cancer_map.items() if key in cancer_type.lower()), None)
    if not study_id: return f"TCGA mapping not found for {cancer_type}."

    try:
        entrez_res = requests.get(f"https://mygene.info/v3/query?q=symbol:{gene_symbol}&fields=entrezgene&species=human").json()
        entrez_id = entrez_res["hits"][0]["entrezgene"]
    except: return f"Could not map {gene_symbol} to Entrez ID."

    base_url = "https://www.cbioportal.org/api"
    sample_list_id = f"{study_id}_all"
    
    try:
        total_samples = len(requests.get(f"{base_url}/studies/{study_id}/samples").json())
        mut_url = f"{base_url}/molecular-profiles/{study_id}_mutations/mutations?entrezGeneId={entrez_id}&sampleListId={sample_list_id}"
        mutated_samples = {m.get("sampleId") for m in requests.get(mut_url).json()} if requests.get(mut_url).status_code == 200 else set()

        cna_url = f"{base_url}/molecular-profiles/{study_id}_cna/discrete-copy-number-alterations?entrezGeneId={entrez_id}&sampleListId={sample_list_id}"
        altered_cna_samples = {c.get("sampleId") for c in requests.get(cna_url).json() if c.get("alteration") in [2, -2]} if requests.get(cna_url).status_code == 200 else set()

        total_altered = len(mutated_samples.union(altered_cna_samples))
        alteration_rate = round((total_altered / total_samples) * 100, 2) if total_samples > 0 else 0

        return f"TCGA POPULATION REALITY CHECK for {gene_symbol}: Altered in {alteration_rate}% of {cancer_type} patients."
    except Exception as e: return f"cBioPortal API Error: {str(e)}"

@st.cache_data(ttl="1d", show_spinner=False)
def check_clinical_survival(gene_symbol: str, cancer_type: str) -> str:
    url = f"https://www.proteinatlas.org/api/search_download.php?search={gene_symbol}&format=json&columns=g,pg&compress=no"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()
        for entry in res:
            if entry.get("Gene", "").upper() == gene_symbol.upper():
                prognostics = entry.get("Pathology prognostics", "")
                if not prognostics: return f"No significant TCGA survival correlation found."
                
                cancer_focus = cancer_type.lower().split()[0] 
                if cancer_focus in prognostics.lower():
                    for p in prognostics.split(","):
                        if cancer_focus in p.lower():
                            if "unfavorable" in p.lower(): return f"🚨 CLINICAL SURVIVAL ALERT: High expression is linked to POORER survival in {cancer_type}."
                            elif "favorable" in p.lower(): return f"🛡️ CLINICAL SURVIVAL ALERT: High expression is linked to BETTER survival in {cancer_type}."
                return f"Survival data exists, but not significantly correlated in {cancer_type}."
        return f"No survival data found."
    except Exception as e: return f"Survival API Error: {str(e)}"

def get_single_cell_artifact_data(gene_symbol: str) -> str:
    id_url = f"https://www.proteinatlas.org/api/search_download.php?search={gene_symbol}&format=json&columns=g,eg&compress=no"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        id_data = requests.get(id_url, headers=headers).json()
        ensembl_id = next((e.get("Ensembl")[0] if isinstance(e.get("Ensembl"), list) else e.get("Ensembl") for e in id_data if e.get("Gene", "").upper() == gene_symbol.upper()), None)
    except: return f"Failed to resolve HPA Ensembl ID."

    if not ensembl_id: return f"No Ensembl ID found."

    try:
        master_data = requests.get(f"https://www.proteinatlas.org/{ensembl_id}.json", headers=headers).json()
        entry = master_data[0] if isinstance(master_data, list) else master_data
        
        cell_types = {}
        for key, val in entry.items():
            if "single cell type specific" in key.lower() and isinstance(val, dict):
                for cell, expr in val.items():
                    try: cell_types[cell.strip()] = float(expr)
                    except ValueError: pass
                break
                
        if not cell_types: return f"No microscopic single-cell data found."
            
        sorted_cells = sorted(cell_types.items(), key=lambda item: item[1], reverse=True)[:5]
        top_cells_str = ", ".join([f"{c[0]} ({c[1]} nTPM)" for c in sorted_cells])
        return f"SINGLE-CELL ARTIFACT CHECK for {gene_symbol}: Predominantly expressed in: {top_cells_str}. WARNING: If these are Macrophages, T-cells, Kupffer cells, or Adipocytes, this is a tissue admixture artifact!"
    except Exception as e: return f"HPA API Error: {str(e)}"

@st.cache_data(show_spinner=False)
def get_validated_antibodies(gene_symbol: str) -> str:
    try:
        id_data = requests.get(f"https://www.proteinatlas.org/api/search_download.php?search={gene_symbol}&format=json&columns=g,eg&compress=no", headers={"User-Agent": "Mozilla/5.0"}).json()
        ensembl_id = next((e.get("Ensembl")[0] if isinstance(e.get("Ensembl"), list) else e.get("Ensembl") for e in id_data if e.get("Gene", "").upper() == gene_symbol.upper()), None)
        if not ensembl_id: return "No antibody data found."

        entry = requests.get(f"https://www.proteinatlas.org/{ensembl_id}.json", headers={"User-Agent": "Mozilla/5.0"}).json()[0]
        antibodies = entry.get("Antibody", [])
        ihc_reliability = entry.get("Reliability (IH)", "Unknown")
        
        if antibodies and ihc_reliability in ["Approved", "Supported", "Enhanced"]:
            ab_list = ", ".join(antibodies) if isinstance(antibodies, list) else antibodies
            return f"HPA-Validated IHC Antibodies: {ab_list} (Reliability: {ihc_reliability})."
        return f"No highly validated IHC antibodies found."
    except: return "Error fetching antibody data."

@st.cache_data(ttl="1d", show_spinner=False)
def fetch_alphafold_structure(uniprot_id):
    try:
        api_res = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}")
        if api_res.status_code == 200:
            data = api_res.json()
            if data and isinstance(data, list):
                file_url = data[0].get("pdbUrl") or data[0].get("cifUrl")
                file_format = "pdb" if data[0].get("pdbUrl") else "cif"
                if file_url and requests.get(file_url).status_code == 200:
                    return requests.get(file_url).text, file_format
        return None, None
    except: return None, None

@st.cache_data(show_spinner=False)
def get_uniprot_binding_sites(uniprot_id):
    try:
        res = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
        if res.status_code == 200:
            target_residues = []
            for f in res.json().get("features", []):
                if f.get("type") in ["Binding site", "Active site"]:
                    start = f.get("location", {}).get("start", {}).get("value")
                    end = f.get("location", {}).get("end", {}).get("value")
                    if start and end: target_residues.extend(list(range(start, end + 1)))
                    elif start: target_residues.append(start)
            return [str(r) for r in set(target_residues)]
        return []
    except: return []

def extract_residue_number(mutation_string):
    match = re.search(r'\d+', str(mutation_string))
    return match.group() if match else None

def render_mutated_protein(structure_data, file_format="pdb", highlight_residues=None):
    view = py3Dmol.view(width=800, height=500)
    view.addModel(structure_data, file_format)
    if highlight_residues:
        view.setStyle({'model': -1}, {"cartoon": {'color': 'lightgrey'}})
        highlight_residues = [str(r) for r in highlight_residues] if isinstance(highlight_residues, list) else [str(highlight_residues)]
        view.addStyle({'resi': highlight_residues}, {'sphere': {'color': 'red', 'radius': 1.2}})
        view.addStyle({'resi': highlight_residues}, {'stick': {'colorscheme': 'blueCarbon'}})
    else:
        view.setStyle({'model': -1}, {"cartoon": {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
    view.zoomTo()
    return view

@st.cache_data(show_spinner=False)
def fetch_visual_network(hugo_symbol, max_nodes=15):
    try:
        res = requests.get(f"https://string-db.org/api/json/network?identifiers={hugo_symbol}&species=9606&limit={max_nodes}")
        return res.json() if res.status_code == 200 else []
    except: return []

def build_pyvis_graph(central_gene, edges_data):
    G = nx.Graph()
    for edge in edges_data:
        node_a, node_b, score = edge.get("preferredName_A"), edge.get("preferredName_B"), edge.get("score", 0)
        if node_a and node_b and score > 0.4: G.add_edge(node_a, node_b, weight=score)

    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="white")
    for node in G.nodes():
        if node == central_gene: net.add_node(node, label=node, color="#EF553B", size=30, shape="star")
        else: net.add_node(node, label=node, color="#636EFA", size=15 + (G.degree(node) * 2))

    for edge in G.edges(data=True): net.add_edge(edge[0], edge[1], value=edge[2]['weight'], color="#4A4A4A")
    net.repulsion(node_distance=200, central_gravity=0.1, spring_length=200, spring_strength=0.05, damping=0.09)
    return net

# --- BENCH-TO-CLOUD WET LAB SCHEMAS ---
class sgRNA(BaseModel):
    target_exon: str = Field(description="Which exon to target (e.g., Exon 2)")
    sequence: str = Field(description="20bp RNA sequence")
    pam: str = Field(description="PAM sequence (e.g., NGG)")
    off_target_risk: str = Field(description="Low, Medium, or High")

class PrimerPair(BaseModel):
    target: str = Field(description="What this primer amplifies")
    forward: str = Field(description="Forward primer sequence (18-22bp)")
    reverse: str = Field(description="Reverse primer sequence (18-22bp)")
    tm: str = Field(description="Melting temperature (e.g., 60 C)")
    amplicon_size: int = Field(description="Expected amplicon size in bp")

class WetLabManifest(BaseModel):
    rationale: str = Field(description="1-sentence explanation.")
    sgrnas: List[sgRNA] = Field(description="Top 2 sgRNA designs.")
    primers: List[PrimerPair] = Field(description="qPCR primers.")

@st.cache_data(show_spinner=False)
def design_validation_experiment(gene_symbol, cancer_type):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.1, api_key=openai_key)
    structured_llm = llm.with_structured_output(WetLabManifest)
    sys_msg = """You are an expert Molecular Biologist and CRISPR designer. Output strict JSON. Sequences MUST be exactly 20 nucleotides + PAM. Primers MUST be 18-22 nucleotides. Include a housekeeping gene control."""
    try:
        response = structured_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=f"Target: {gene_symbol}\nCancer: {cancer_type}")])
        return response.model_dump()
    except: return None

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def planner_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0, api_key=openai_key)
    structured_llm = llm.with_structured_output(Plan)
    sys_msg = "You are an expert Clinical Bioinformatics Planner. Output a step-by-step plan to gather data using OpenTargets, OncoKB, PubMed, and ClinicalTrials."
    response = structured_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=f"Prompt: {state.get('user_prompt')}\nGenes: {state.get('significant_genes')}")])
    return {"plan": response.steps}

def fast_triage_node(state: AgentState):
    st.markdown("⚡ **[NODE: Fast Triage]** Running the high-speed gauntlet...")
    genes = state.get("significant_genes", [])
    intent = state.get("biomarker_intent", "Therapeutic")
    triage_results = []
    for gene_info in genes:
        hugo, tumor_type = gene_info.get("hugo"), gene_info.get("tumor_type")
        st.markdown(f"      -> Running fast APIs for {hugo}...")
        triage_data = {
            "gene": hugo,
            "biology": get_gene_info(hugo).get('summary', 'No summary.'),
            "tissue_gtex": fetch_normal_tissue_profile(hugo),
            "hpa_single_cell": get_single_cell_artifact_data(hugo),
            "tcga_freq": get_tcga_population_frequency(hugo, tumor_type)
        }
        if "Diagnostic" in intent: triage_data["detectability"] = check_biomarker_detectability(hugo)
        else: triage_data["open_targets"] = fetch_target_tractability(hugo)
        triage_results.append(triage_data)
    return {"fast_triage_data": triage_results}

class SelectionResult(BaseModel):
    evaluations: List[KnowledgePath] = Field(description="The evaluation path for every single gene processed.")
    top_candidates: List[str] = Field(description="List of Hugo symbols chosen for the Deep Dive.")

def intelligent_selection_node(state: AgentState):
    st.markdown("⚖️ **[NODE: Intelligent Selection]** AI grading Knowledge Paths to draft top candidates...")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0, api_key=openai_key)
    structured_llm = llm.with_structured_output(SelectionResult)
    
    triage_data = state.get("fast_triage_data", [])
    intent, modality = state.get("biomarker_intent", "Therapeutic"), state.get("therapeutic_modality", "Small Molecule / Kinase Inhibitor")
    max_limit = state.get("max_deep_dive", 5)
    
    modality_rules = "- For Therapeutics: Prioritize high tractability and high DepMap essentiality."
    if modality == "CAR-T / ADC / Radioligand":
        modality_rules = "- CAR-T/ADC STRICT ROUTING: IGNORE DepMap essentiality. Mandate Cell Surface Localization (Membrane). Penalize ubiquitous normal tissue expression."
    
    sys_msg = f"""You are the Lead Bioinformatics Architect. Evaluate candidates for a {intent} pipeline. Return up to {max_limit} absolute best candidates.
    If 'hpa_single_cell' indicates Macrophages, T-cells, Kupffer cells, or Adipocytes, DO NOT automatically exclude. Evaluate if it's a TME Biomarker.
    {modality_rules}
    OUTPUT FORMAT (DR.KNOWS): Provide a reasoning path for EVERY gene. Example: '[HPA: Macrophage] -> INCLUDE (TME Marker)'"""
    
    response = structured_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=json.dumps(triage_data))])
    top_cands = [c.strip().upper() for c in response.top_candidates]
    original_genes = state.get("significant_genes", [])
    winning_genes = [g for g in original_genes if str(g["hugo"]).strip().upper() in top_cands]
    
    logic_str = "\n".join([f"- **{p.gene}**: {p.path}" for p in response.evaluations])
    if not winning_genes:
        winning_genes = original_genes[:max_limit]
        logic_str += f"\n\n⚠️ **SYSTEM OVERRIDE:** Forcing top {max_limit} targets to prevent starvation."
    st.markdown(f"### 🧠 AI Funnel Reasoning\n{logic_str}")
    return {"significant_genes": winning_genes, "selection_logic": logic_str}

def executor_node(state: AgentState):
    plan_text = " ".join(state.get("plan", [])).lower()
    winning_genes = state.get("significant_genes", []) 
    fast_data = state.get("fast_triage_data", [])
    intent = state.get("biomarker_intent", "Therapeutic")
    new_evidence = []
    
    for gene_info in winning_genes:
        hugo, alt, tumor_type = gene_info.get("hugo"), gene_info.get("alteration"), gene_info.get("tumor_type")
        fast_dossier = next((item for item in fast_data if item["gene"] == hugo), {})
        
        report = {"gene": hugo, "alteration": alt, "source": gene_info.get("source", ""), "biology": get_gene_info(hugo), "evidence": {}}
        report["evidence"]["TCGA_Frequency"] = fast_dossier.get("tcga_freq", "Unknown")
        report["evidence"]["HPA_SingleCell"] = fast_dossier.get("hpa_single_cell", "Unknown")
        if "Diagnostic" in intent: report["evidence"]["Detectability"] = fast_dossier.get("detectability", "Unknown")
        else: report["evidence"]["OpenTargets"] = fast_dossier.get("open_targets", {})

        st.markdown(f"      -> Hunting UniProt for Active Sites for {hugo}...")
        uniprot_id = get_uniprot_id(hugo)
        if uniprot_id: report["evidence"]["UniProt_Pockets"] = {"has_defined_pockets": len(get_uniprot_binding_sites(uniprot_id)) > 0}
            
        st.markdown(f"      -> Extracting validated IHC Antibodies for {hugo}...")
        report["evidence"]["IHC_Antibodies"] = get_validated_antibodies(hugo)

        st.markdown(f"      -> Querying TCGA Kaplan-Meier Survival Outcomes for {hugo}...")
        report["evidence"]["Survival_Outcomes"] = check_clinical_survival(hugo, tumor_type)
        
        if "oncokb" in plan_text: report["evidence"]["OncoKB"] = get_onco_data(hugo, alt, tumor_type)
            
        if "Discovery" in state.get("analysis_mode", "Clinical Triage"):
            st.markdown(f"      -> Fetching STRING protein network for {hugo}...")
            report["evidence"]["STRING_Interactions"] = get_protein_interactions(hugo)
            
        if "pubmed" in plan_text:
            interactors = report.get("evidence", {}).get("STRING_Interactions", {}).get("interacting_proteins", [])
            pubmed_data = search_pubmed(hugo, tumor_type, mode=state.get("analysis_mode", "Clinical Triage"), intent=intent, interactors=interactors)
            report["evidence"]["PubMed_Provenance"] = pubmed_data.get("provenance", [])
            
            if pubmed_data.get("status") == "Success" and pubmed_data.get("papers"):
                st.markdown(f"   -> Grading literature relevance for {hugo}...")
                grader_llm = ChatOpenAI(model="gpt-5.2", temperature=0, api_key=openai_key).with_structured_output(PaperScore)
                candidate_papers = pubmed_data["papers"]
                good_papers = []
                
                for p in candidate_papers:
                    if len(good_papers) >= 3: break
                    eval_prompt = f"""Evaluate abstract relevance to target {hugo}. 
                    CRITICAL RUBRIC:
                    - Score 1-3: Acronym collision, unrelated disease.
                    - Score 4-10: Relevant context found.
                    Title: {p['Title']} Abstract: {p['Abstract'][:800]}"""
                    try:
                        score_result = grader_llm.invoke([SystemMessage(content="Output strict JSON grading."), HumanMessage(content=eval_prompt)])
                        p["AI_Score"], p["AI_Reason"] = score_result.score, score_result.reason
                        
                        if score_result.score >= 4: # <-- TUNED DOWN FROM 5
                            good_papers.append(p)
                        else:
                            ai_filtered_evidence = state.get("ai_filtered_evidence", [])
                            ai_filtered_evidence.append({"Gene": hugo, "Score": score_result.score, "Reason": score_result.reason, "Title": p["Title"], "PMID": p["PMID"]})
                            state["ai_filtered_evidence"] = ai_filtered_evidence
                    except:
                        p["AI_Score"], p["AI_Reason"] = "?", "Error"
                        good_papers.append(p) 
                pubmed_data["papers"] = good_papers
            report["evidence"]["PubMed"] = pubmed_data

        if "clinicaltrials" in plan_text or "trials" in plan_text:
            st.markdown(f"   -> Fetching Clinical Trials for {hugo}...")
            report["evidence"]["ClinicalTrials"] = search_clinical_trials(hugo, tumor_type)
            
        new_evidence.append(report)
        
    return {"gathered_evidence": new_evidence, "pathway_data": state.get("pathway_data"), "ai_filtered_evidence": state.get("ai_filtered_evidence", [])}

def clinical_review_node(state: AgentState):
    st.markdown("🧑‍⚕️ **[NODE: Clinical Review]** Experts are debating the evidence...")
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.2, api_key=openai_key)
    
    prompt = f"""You are hosting a clinical tumor board. Review this data for {state.get('user_prompt')}.
    Evidence: {json.dumps(state.get('gathered_evidence'))}
    
    Speak as a MOLECULAR PATHOLOGIST to evaluate tissue context, a CLINICAL CHEMIST/ONCOLOGIST to evaluate detectability/druggability, and a BIOINFORMATICS AUDITOR to evaluate literature collisions.
    
    CRITICAL FORMATTING RULE: Do NOT use Heading 1 (#) or Heading 2 (##) anywhere in your response. Only use Heading 4 (####) or bold text (**) for section titles. Do not make the text massive.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"expert_consensus": response.content}

def writer_node(state: AgentState):
    st.markdown("✍️ **[NODE: Writer]** Synthesizing the final clinical report...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_key)

    # --- 1. THE NEW TME MATH CONTEXT ---
    tme_data = state.get("tme_deconvolution", {})
    tme_context_str = "TME Deconvolution was not run by the user."
    
    if tme_data and not tme_data.get("error"):
        all_fractions = pd.DataFrame([d["fractions"] for d in tme_data["metrics"].values()])
        avg_r2 = np.mean([d["r_squared"] for d in tme_data["metrics"].values()])
        tme_summary = all_fractions.describe().T
        
        tme_context_str = f"Cohort TME Profile (Algorithm Fit R-Squared: {avg_r2:.2f}):\n"
        for cell in tme_summary.index:
            max_val = tme_summary.loc[cell, 'max']
            if max_val > 5.0:  # Normalized to 100 scale!
                tme_context_str += f"- {cell}: Mean {tme_summary.loc[cell, 'mean']:.1f}% (Peak Heterogeneity Max: {max_val:.1f}%)\n"

    intent = state.get("biomarker_intent", "Therapeutic Target (Drug Discovery)")
    
    # --- 2. ASSEMBLE THE DATA PAYLOAD ---
    winning_genes = [g.get('gene') for g in state.get('gathered_evidence', [])]
    winning_genes_str = ", ".join(winning_genes) if winning_genes else "None"

    user_context = f"""
    Disease Target: {state.get('user_prompt')}
    
    IMPORTANT CONTEXT: The AI Triage Funnel evaluated a large list of dysregulated genes, but strictly selected ONLY the following candidates for Deep-Dive Evidence Gathering: [{winning_genes_str}]. 
    You MUST focus your report EXCLUSIVELY on these surviving candidates based on the evidence below. Do not mention missing evidence for other genes.
    
    TME Profile: {tme_context_str}
    Pathway Data: {json.dumps(state.get('pathway_data', {}))}
    Expert Consensus (Tumor Board): {state.get('expert_consensus')}
    Gathered Evidence: {json.dumps(state.get('gathered_evidence'))}
    """

    # --- 3. THE OLD "GOLD STANDARD" PROMPT + NEW TME RULES ---
    modality = state.get("therapeutic_modality", "Small Molecule / Kinase Inhibitor")
    sys_msg = f"""You are an expert Systems Biologist and Medical Writer.
    Write a pathway-centric scientific report evaluating these genes as drug targets for: {modality}.
    
    CRITICAL GUARDRAILS:
    1. TONE AND STYLE: Write confidently as if authoring a published review article.
    2. THE ARTIFACT KILLER & TME MAPPING: Review the following cohort TME profile:
    {tme_context_str}
    If the HPA single-cell data flags a gene as belonging to Macrophages, Kupffer cells, T-cells, or Adipocytes, AND that cell type shows high variance or abundance in the TME profile, explicitly state that the gene is a Tissue Admixture Artifact, not a tumor-intrinsic driver. Place it in Tier 4.
    
    YOU MUST STRICTLY USE THE EXACT MARKDOWN TEMPLATE BELOW. DO NOT DEVIATE:
    
    ## 📊 Executive Summary
    [3-4 sentences explaining target selection. Explicitly mention the Tumor Microenvironment composition.]
    
    ## 🕸️ Systems Biology & Pathway Dysregulation
    [Synthesis of the KEGG pathway data]
    
    ## 🔬 Targetable Hubs & Translational Risk Tiers
    [Categorize EACH evaluated gene into its appropriate Tier. Then, for EVERY gene, you MUST provide the following breakdown:]
    
    ### [Gene Symbol] - [Tier Assignment]
    * **🧬 Pathology & Tissue Context:** [Extract the EXACT arguments from the Molecular Pathologist in the Expert Consensus. Explicitly mention the HPA single-cell data and TME admixture risks.]
    * **💊 Oncology & Actionability:** [Extract the EXACT arguments from the Medical Oncologist/Clinical Chemist. Discuss OpenTargets tractability or DepMap essentiality.]
    * **📚 Literature Verdict:** [Summarize the Bioinformatics Auditor's findings on PubMed collisions or relevant trials.]
    
    ## 🏥 Translational Outlook
    [Summarize relevant clinical trials and approved drugs]
    
    ### 🧪 Recommended Next Experimental Steps
    [Provide 3-4 bullet points. Provide specific HPA antibody catalog numbers for validation.]
    """

    response = llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_context)
    ])
    
    st.markdown("✅ **Final report successfully written.**")
    return {"final_report": response.content}

# --- LANGGRAPH COMPILATION ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("fast_triage", fast_triage_node)
workflow.add_node("intelligent_selection", intelligent_selection_node)
workflow.add_node("executor", executor_node)
workflow.add_node("clinical_review", clinical_review_node) 
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "fast_triage")
workflow.add_edge("fast_triage", "intelligent_selection")
workflow.add_edge("intelligent_selection", "executor")
workflow.add_edge("executor", "clinical_review")
workflow.add_edge("clinical_review", "writer")
workflow.add_edge("writer", END)
orchestrator = workflow.compile()

# ==========================================
# STREAMLIT FRONTEND & UI
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Session Intention & Data")
    user_intention = st.text_area("Research Goal / Intention (Optional)", placeholder="E.g., 'Focus on resistance mechanisms...'")
    counts_file = st.file_uploader("Upload RNA Counts (CSV)", type=["csv"])
    metadata_file = st.file_uploader("Upload Metadata (CSV)", type=["csv"])
    
    st.markdown("---")
    st.subheader("Optional: DNA Mutational Profile")
    dna_file = st.file_uploader("Upload DNA Variants (CSV)", type=["csv"])
    
    condition_col, batch_col = "condition", "None"
    
    if metadata_file is not None:
        temp_meta = pd.read_csv(metadata_file, index_col=0, nrows=0) 
        meta_cols = temp_meta.columns.tolist()
        metadata_file.seek(0)
        
        st.markdown("---")
        st.subheader("2. Experimental Design")
        col_a, col_b = st.columns(2)
        with col_a: condition_col = st.selectbox("Primary Contrast", meta_cols, index=meta_cols.index("condition") if "condition" in meta_cols else 0)
        with col_b: batch_col = st.selectbox("Batch Covariate", ["None"] + meta_cols)
            
    st.markdown("---")
    st.subheader("4. Clinical Context & AI Triage")
    with st.form("clinical_triage_form"):
        cancer_type = st.text_input("Cancer Type (e.g., Melanoma, NSCLC)", value="Melanoma")
        biomarker_intent = st.radio("Biomarker Goal", ["Therapeutic Target (Drug Discovery)", "Diagnostic/Risk Biomarker (Screening/Monitoring)"])
        therapeutic_modality = st.selectbox("Therapeutic Modality", ["Small Molecule / Kinase Inhibitor", "CAR-T / ADC / Radioligand"])
        analysis_mode = st.radio("Analysis Mode", ["Clinical Triage (Known Targets)", "Biomarker Discovery (Novel Targets)"])
        
        st.markdown("#### 🎯 Target Selection Funnel")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1: n_up_pathway = st.number_input("Upregulated Drivers", min_value=0, max_value=30, value=10)
        with col_r2: n_down_pathway = st.number_input("Downregulated", min_value=0, max_value=30, value=5)
        with col_r3: n_outliers = st.number_input("Outliers", min_value=0, max_value=30, value=5)
        n_deep_dive = st.slider("Final Candidates (Max)", min_value=1, max_value=10, value=3)
        
        save_clinical_btn = st.form_submit_button("💾 Save Context", type="primary")

    if save_clinical_btn:
        st.session_state.clinical_context = {
            "cancer_type": cancer_type, "biomarker_intent": biomarker_intent,
            "therapeutic_modality": therapeutic_modality, "analysis_mode": analysis_mode,
            "n_up_pathway": n_up_pathway, "n_down_pathway": n_down_pathway,
            "n_outliers": n_outliers, "n_deep_dive": n_deep_dive,
            "top_n_genes": n_up_pathway + n_down_pathway + n_outliers
        }
        st.success(f"✅ Context Locked! You may now run the AI Pipeline.")
    
    st.markdown("### 🧑‍⚕️ Clinical Safety & Evidence")
    hitl_toggle = st.toggle("⏸️ Enable Human-in-the-Loop", value=True)
    baseline_toggle = st.toggle("⚖️ Head-to-Head Baseline", value=False)
    
    # --- UNIFIED EXECUTION BUTTON ---
    run_button = st.button("🚀 Run AI Pipeline (Execute LangGraph)", width="stretch", type="primary")
    
    st.markdown("---")
    st.subheader("5. Custom Knowledge (Optional)")
    uploaded_pdf = st.file_uploader("Upload Lab Protocols/Guidelines (PDF)", type=["pdf"])

with col2:
    st.subheader("1. PCA Quality Control (QC) Gate")
    pca_container = st.container() 
    
    st.markdown("---")
    st.subheader("2. Tumor Microenvironment (TME) Deconvolution")
    tme_container = st.container() 
    
    st.markdown("---")
    st.subheader("3. Statistical Cutoffs")
    with st.form("stats_form"):
        de_engine = st.selectbox("Differential Expression Engine", ["PyDESeq2", "EdgePy", "RPKM/T-Test"])
        pval_thresh = st.number_input("P-Value Cutoff", min_value=0.0001, max_value=0.1000, value=0.0500)
        log2fc_thresh = st.slider("Log2FC Threshold", min_value=0.0, max_value=10.0, value=2.0)
        update_plot_btn = st.form_submit_button("📊 Generate Volcano Plot")

    if counts_file and metadata_file:
        counts_df_raw = pd.read_csv(counts_file, index_col=0)
        metadata_df_raw = pd.read_csv(metadata_file, index_col=0)
        
        with pca_container:
            st.info("Visually inspect your samples before running differential expression.")
            with st.spinner("Calculating Principal Components..."):
                pca_data = counts_df_raw  
                scaled_data = StandardScaler().fit_transform(pca_data)
                
                # 1. Calculate PCA and extract the variance ratio
                pca_model = PCA(n_components=2)
                pca_results = pca_model.fit_transform(scaled_data)
                var_ratio = pca_model.explained_variance_ratio_ * 100 # Convert to percentage
                
                pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'], index=pca_data.index).join(metadata_df_raw, how='inner').reset_index() 
                plot_symbol = batch_col if batch_col != "None" and batch_col in pca_df.columns else None
                
                # 2. Inject variance into the axis labels
                pca_fig = px.scatter(
                    pca_df, x='PC1', y='PC2', 
                    color=condition_col if condition_col in pca_df.columns else None, 
                    symbol=plot_symbol, hover_name=pca_df.columns[0],
                    labels={
                        "PC1": f"PC1 ({var_ratio[0]:.1f}%)", 
                        "PC2": f"PC2 ({var_ratio[1]:.1f}%)"
                    },
                    title="Patient Sample Clustering (PCA)"
                )
                st.plotly_chart(pca_fig, width="stretch")

        with tme_container:
            if 'tme_analysis_complete' not in st.session_state:
                st.session_state.tme_analysis_complete = False
                st.session_state.tme_results = None
                st.session_state.tme_stats = None

            if st.button("🧬 Run Rigorous TME Analysis", type="primary", width="stretch"):
                with st.spinner("Deconvolving TME Axes..."):
                    try:
                        from tme_core import TMECore
                        engine = TMECore(counts_df_raw, metadata_df_raw)
                        results, stats = engine.run_analysis()
                        st.session_state.tme_results = results
                        st.session_state.tme_stats = stats
                        st.session_state.tme_analysis_complete = True
                    except Exception as e:
                        st.error(f"❌ TME Engine Error: {str(e)}")

            if st.session_state.get('tme_analysis_complete'):
                from dashboard_components import render_tme_dashboard
                key_finding, risk_col_name = render_tme_dashboard(st.session_state.tme_results, st.session_state.tme_stats)

        # --- VOLCANO PLOT ---
        if update_plot_btn:
            counts_file.seek(0)
            metadata_file.seek(0)
            counts_df = pd.read_csv(counts_file, index_col=0)
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            
            with st.spinner(f"Calculating Differential Expression using {de_engine}..."):
                design_factors = [batch_col, condition_col] if batch_col != "None" and batch_col != condition_col else condition_col
                unique_levels = metadata_df[condition_col].dropna().unique()
                level_1 = unique_levels[0]
                level_2 = unique_levels[1] if len(unique_levels) > 1 else unique_levels[0]
                
                if st.session_state.get("use_mock_mode", False):
                    output = run_differential_stats(counts_df, metadata_df, condition_col, level_1, level_2, mock_mode=True)
                    results_df = output["results_df"]
                else:
                    if de_engine == "PyDESeq2":
                        dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors=design_factors)
                        dds.deseq2()
                        stat_res = DeseqStats(dds, contrast=[condition_col, level_1, level_2])
                        stat_res.summary()
                        results_df = stat_res.results_df
                    elif de_engine == "RPKM/T-Test":
                        output = run_differential_stats(counts_df, metadata_df, condition_col, level_1, level_2, mock_mode=False)
                        results_df = output["results_df"]
            
            plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
            plot_df['-log10(padj)'] = -np.log10(plot_df['padj'] + 1e-300)
            conditions = [
                (plot_df['padj'] < pval_thresh) & (plot_df['log2FoldChange'] > log2fc_thresh), 
                (plot_df['padj'] < pval_thresh) & (plot_df['log2FoldChange'] < -log2fc_thresh)
            ]
            plot_df['Significance'] = np.select(conditions, ['Upregulated', 'Downregulated'], default='Not Significant')
            
            st.session_state.upregulated_df = plot_df[plot_df['Significance'] == 'Upregulated'].sort_values(by='padj')
            st.session_state.full_results_df = plot_df.copy()

            fig = px.scatter(
                plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', 
                color_discrete_map={'Upregulated': '#EF553B', 'Downregulated': '#636EFA', 'Not Significant': '#4A4A4A'},
                hover_name=plot_df.index, render_mode='webgl', title="Gene Expression Volcano Plot"
            )
            fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash", line_color="white")
            fig.add_vline(x=log2fc_thresh, line_dash="dash", line_color="white")
            fig.add_vline(x=-log2fc_thresh, line_dash="dash", line_color="white")
            st.session_state.volcano_fig = fig
            
        if st.session_state.volcano_fig:
            st.plotly_chart(st.session_state.volcano_fig, width="stretch")
            if len(st.session_state.ai_targets) > 0:
                st.success(f"✅ **{len(st.session_state.ai_targets)} Targets active.**")

# ==========================================
# UNIFIED AI EXECUTION BLOCK
# ==========================================
if run_button and counts_file and metadata_file:
    st.markdown("---")
    st.subheader("🤖 AI Clinical Orchestration")
    
    if not st.session_state.get('clinical_context'):
        st.error("⚠️ Please click 'Save Context' in Section 4 first to lock in your clinical parameters.")
        st.stop()
    if 'full_results_df' not in st.session_state or st.session_state.full_results_df.empty:
        st.error("⚠️ Please generate the Volcano Plot first to supply the math engine outputs.")
        st.stop()
        
    ctx = st.session_state.clinical_context
    ACTIONABLE_GENES = ["BRAF", "EGFR", "KRAS", "PIK3CA", "ERBB2", "ALK", "BRCA1", "BRCA2", "TP53"]
    
    with st.spinner("🧠 Recruiting Hybrid Target Roster via Local GSEA..."):
        full_df = st.session_state.full_results_df
        gsea_input_df = full_df[(~full_df.index.isin(ACTIONABLE_GENES)) & (~full_df.index.str.match(r'^(RPL|RPS|MT-)'))] if "Discovery" in ctx["analysis_mode"] else full_df
        
        up_df_pool = gsea_input_df[gsea_input_df['log2FoldChange'] > 0].sort_values(by='padj')
        down_df_pool = gsea_input_df[gsea_input_df['log2FoldChange'] < 0].sort_values(by='padj')
        extreme_df_pool = gsea_input_df.sort_values(by='padj')

        pathway_results = run_gsea_analysis(gsea_input_df)
        st.session_state.gsea_obj = pathway_results.pop("gsea_obj", None)
        
        cluster_targets, roster_metadata = [], []
        up_pathways = [pw for pw in pathway_results.get("pathways", []) if pw.get("nes", 0) > 0]
        down_pathways = [pw for pw in pathway_results.get("pathways", []) if pw.get("nes", 0) < 0]

        up_count = 0
        for pw in up_pathways:
            for g in pw["overlapping_genes"]:
                if up_count >= ctx["n_up_pathway"]: break
                if g in up_df_pool.index and g not in cluster_targets:
                    cluster_targets.append(g)
                    roster_metadata.append({"gene": g, "source": f"Upregulated Driver ({pw['pathway']})", "alteration": "Overexpressed"})
                    up_count += 1
                    
        down_count = 0
        for pw in down_pathways:
            for g in pw["overlapping_genes"]:
                if down_count >= ctx["n_down_pathway"]: break
                if g in down_df_pool.index and g not in cluster_targets:
                    cluster_targets.append(g)
                    roster_metadata.append({"gene": g, "source": f"Downregulated Biomarker ({pw['pathway']})", "alteration": "Loss of Expression"})
                    down_count += 1

        outlier_count = 0
        for g in extreme_df_pool.index:
            if outlier_count >= ctx["n_outliers"]: break
            if g not in cluster_targets:
                cluster_targets.append(g)
                direction = "Overexpressed" if full_df.loc[g, 'log2FoldChange'] > 0 else "Loss of Expression"
                roster_metadata.append({"gene": g, "source": "Lone Wolf (Statistical Outlier)", "alteration": direction})
                outlier_count += 1

        st.session_state.ai_targets = cluster_targets
        st.session_state.roster_metadata = roster_metadata

    rag_context = ""
    if uploaded_pdf is not None:
        with st.spinner("📚 Reading Lab Protocol..."):
            vectorstore = process_pdf_for_rag(uploaded_pdf)
            if vectorstore:
                docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(f"Context for {ctx['cancer_type']}")
                rag_context = "\n\n".join([d.page_content for d in docs])

    with st.status("🧠 Live Agent Thought Trace (Glass Box)", expanded=True) as status:
        structured_genes, dna_gene_names = [], []
        
        if dna_file is not None:
            dna_df = pd.read_csv(dna_file)
            if 'Gene' in dna_df.columns and 'Alteration' in dna_df.columns:
                for _, row in dna_df.iterrows():
                    dna_gene_names.append(str(row['Gene']).strip())
                    structured_genes.append({"hugo": str(row['Gene']).strip(), "alteration": str(row['Alteration']).strip(), "tumor_type": ctx["cancer_type"], "source": "DNA Mutation"})
        
        for target in st.session_state.roster_metadata:
            structured_genes.append({"hugo": target["gene"], "alteration": target["alteration"], "tumor_type": ctx["cancer_type"], "source": target["source"]})
            
        base_task = f"Analyze dysregulated genes ({', '.join(st.session_state.ai_targets)}) in {ctx['cancer_type']} focusing on {ctx['biomarker_intent']}."
        prompt_text = f"USER'S INTENTION: '{user_intention.strip()}'\n\nCORE TASK: {base_task}" if user_intention.strip() else base_task

        # --- THE FIX: MAP TME DATA CORRECTLY TO INITIAL STATE ---
        tme_payload = {"error": True, "metrics": {}}
        if st.session_state.get('tme_analysis_complete') and st.session_state.get('tme_results') is not None:
            tme_dict = {}
            for idx, row in st.session_state.tme_results.iterrows():
                # 1. Extract only the numeric cell fractions
                raw_fractions = {k: float(v) for k, v in row.items() if k not in ['Sample_ID', 'R2', 'risk_category'] and isinstance(v, (int, float))}
                
                # 2. FORCE NORMALIZATION: Ensure they sum to exactly 1.0 (100%)
                total_weight = sum(raw_fractions.values())
                if total_weight > 0:
                    norm_fractions = {k: (v / total_weight) for k, v in raw_fractions.items()}
                else:
                    norm_fractions = raw_fractions # Fallback if all weights are 0
                
                tme_dict[row['Sample_ID']] = {"fractions": norm_fractions, "r_squared": row.get('R2', 0.8)}
            tme_payload = {"error": False, "metrics": tme_dict}

        initial_state = {
            "user_prompt": prompt_text,
            "significant_genes": structured_genes,
            "plan": [], "gathered_evidence": [],
            "pathway_data": pathway_results, 
            "final_report": "", "custom_knowledge": rag_context, 
            "analysis_mode": ctx["analysis_mode"],
            "biomarker_intent": ctx["biomarker_intent"],  
            "therapeutic_modality": ctx["therapeutic_modality"], 
            "max_deep_dive": ctx["n_deep_dive"],          
            "fast_triage_data": [], "selection_logic": "",                
            "discarded_evidence": [], "ai_filtered_evidence": [],
            "expert_consensus": "",
            "tme_deconvolution": tme_payload # FIXED!
        }
        
        st.session_state.run_baseline = baseline_toggle
        st.session_state.base_cancer_type = ctx["cancer_type"]
        st.session_state.base_prompt = prompt_text
        st.session_state.agent_state = initial_state
        
        with get_openai_callback() as cb:
            st.session_state.agent_state.update(planner_node(st.session_state.agent_state))
            st.session_state.agent_state.update(fast_triage_node(st.session_state.agent_state))
            st.session_state.agent_state.update(intelligent_selection_node(st.session_state.agent_state))
            st.session_state.agent_state.update(executor_node(st.session_state.agent_state))
            
            st.session_state.total_tokens += cb.total_tokens
            st.session_state.total_cost += cb.total_cost
        
        st.session_state.gathering_complete = True
        st.session_state.run_complete = False 
        
        if not hitl_toggle:
            st.session_state.agent_state.update(clinical_review_node(st.session_state.agent_state)) 
            st.session_state.agent_state.update(writer_node(st.session_state.agent_state))
            st.session_state.run_complete = True
            st.session_state.final_report = st.session_state.agent_state["final_report"]
            st.session_state.plan = st.session_state.agent_state["plan"]
            st.session_state.pathway_data = st.session_state.agent_state.get("pathway_data", {})
        st.rerun()

# --- PHASE 1.5: THE HUMAN-IN-THE-LOOP PAUSE ---
if st.session_state.get("gathering_complete") and not st.session_state.get("run_complete") and hitl_toggle:
    st.markdown("---")
    st.subheader("⏸️ Human-in-the-Loop: Review Evidence")
    
    flat_papers = []
    for g_idx, g_data in enumerate(st.session_state.agent_state.get("gathered_evidence", [])):
        for p_idx, p in enumerate(g_data.get("evidence", {}).get("PubMed", {}).get("papers", [])):
            flat_papers.append({"Keep": True, "Score": p.get("AI_Score", "?"), "AI Reason": p.get("AI_Reason", "N/A"), "Gene": g_data["gene"], "Title": p["Title"], "PMID": p["PMID"], "_g_idx": g_idx, "_p_idx": p_idx})
            
    if flat_papers:
        edited_df = st.data_editor(pd.DataFrame(flat_papers)[["Keep", "Score", "AI Reason", "Gene", "Title", "PMID"]], hide_index=True, width="stretch", disabled=["Score", "AI Reason", "Gene", "PMID", "Title"])
    else:
        st.info("💡 No experimental literature passed the AI filter. Synthesis will rely on Pathway & Systems Biology.")
        edited_df = pd.DataFrame()

    if st.button("🚀 Step 2: Approve Evidence & Synthesize Report", type="primary", width="stretch"):
        with st.spinner("✍️ Synthesizing the final clinical report..."):
            approved_evidence = copy.deepcopy(st.session_state.agent_state["gathered_evidence"])
            discarded_papers = [] 
            
            if not edited_df.empty:
                for g_data in approved_evidence:
                    if "PubMed" in g_data.get("evidence", {}) and "papers" in g_data["evidence"]["PubMed"]:
                        g_data["evidence"]["PubMed"]["papers"] = []
                for i, row in edited_df.iterrows():
                    g_idx, p_idx = flat_papers[i]["_g_idx"], flat_papers[i]["_p_idx"]
                    orig_paper = st.session_state.agent_state["gathered_evidence"][g_idx]["evidence"]["PubMed"]["papers"][p_idx]
                    if row["Keep"]: approved_evidence[g_idx]["evidence"]["PubMed"]["papers"].append(orig_paper)
                    else: discarded_papers.append({"Gene": flat_papers[i]["Gene"], "Title": orig_paper.get("Title"), "PMID": orig_paper.get("PMID")})
                        
            st.session_state.agent_state["gathered_evidence"] = approved_evidence
            st.session_state.agent_state["discarded_evidence"] = discarded_papers
            
            with get_openai_callback() as cb:
                st.session_state.agent_state.update(clinical_review_node(st.session_state.agent_state))
                st.session_state.agent_state.update(writer_node(st.session_state.agent_state))
                st.session_state.total_tokens += cb.total_tokens
                st.session_state.total_cost += cb.total_cost

            st.session_state.run_complete = True
            st.session_state.final_report = st.session_state.agent_state["final_report"]
            st.session_state.plan = st.session_state.agent_state["plan"]
            st.session_state.pathway_data = st.session_state.agent_state.get("pathway_data", {})
        st.rerun()

# ==========================================
# 6. RENDER RESULTS, VISUALS & CHATBOT
# ==========================================
if st.session_state.run_complete:
    
    # --- RESTORED: THE GLASS BOX (Plan & Funnel Logic) ---
    st.markdown("### 🔍 AI Thought Trace & Funnel Logic")
    
    plan = st.session_state.agent_state.get("plan", [])
    if plan:
        with st.expander("📋 View the AI's Strategic Plan"):
            st.info("This is the step-by-step roadmap the Planner Agent generated before executing the tool calls.")
            for step in plan:
                st.write(f"- {step}")

    selection_logic = st.session_state.agent_state.get("selection_logic", "")
    if selection_logic:
        with st.expander("⚖️ View AI Funnel Selection Logic (DR.KNOWS)"):
            st.info("The AI evaluated all targets from the Wide Net using fast APIs and drafted the top candidates based on these logic paths:")
            st.markdown(selection_logic)
            
    st.markdown("---")
    st.subheader("📄 Final Synthesized Clinical Report")
    st.markdown(st.session_state.final_report)
    
    st.markdown("### 🔍 AI Thought Trace & Funnel Logic")
    
    plan = st.session_state.agent_state.get("plan", [])
    if plan:
        with st.expander("📋 View the AI's Strategic Plan", expanded=True):
            st.info("This is the step-by-step roadmap the Planner Agent generated before executing the tool calls.")
            for step in plan:
                st.write(f"- {step}")

    selection_logic = st.session_state.agent_state.get("selection_logic", "")
    if selection_logic:
        with st.expander("⚖️ View AI Funnel Selection Logic (DR.KNOWS)", expanded=True):
            st.info("The AI evaluated all targets from the Wide Net using fast APIs and drafted the top candidates based on these logic paths:")
            st.markdown(selection_logic)

    consensus = st.session_state.agent_state.get("expert_consensus", "")
    if consensus:
        with st.expander("🧑‍⚕️ View Raw Tumor Board Debate (Pathologist vs. Oncologist)"):
            st.markdown(consensus)
            
    st.markdown("### 📚 Reference Library & Evidence Audit")
    has_kept_papers = False
    with st.expander("✅ PubMed Literature Included in Synthesis"):
        for g_data in st.session_state.agent_state.get("gathered_evidence", []):
            papers = g_data.get("evidence", {}).get("PubMed", {}).get("papers", [])
            if papers:
                has_kept_papers = True
                st.markdown(f"**{g_data['gene']}**")
                for p in papers: st.markdown(f"- **PMID {p['PMID']}**: *{p['Title']}*")
        if not has_kept_papers: st.info("No literature passed filters. Systems biology used.")
        # 3. Show the Papers that the AI threw out (The Acronym Catcher)
    ai_discarded = st.session_state.agent_state.get("ai_filtered_evidence", [])
    if ai_discarded:
        with st.expander("🤖 AI Pre-Filtered Literature (Auto-Discarded)", expanded=True):
            st.info("The AI evaluated up to 10 papers per gene. The following papers scored < 4 and were automatically excluded to prevent hallucinations and acronym collisions.")
            for doc in ai_discarded:
                st.markdown(f"- **{doc['Gene']}** (Score: {doc['Score']}): *{doc['Title']}* - Reason: `{doc['Reason']}`")

    st.markdown("---")
    st.subheader("📊 Multi-Modal Target Analytics")
    if st.session_state.get("ai_targets"):
        viz_target = st.selectbox("🎯 Select Target to Analyze:", st.session_state.ai_targets)
        if st.button("Generate Dual Visualization", type="primary", width="stretch"):
            col_net, col_struct = st.columns(2)
            with col_net:
                st.markdown(f"#### 🕸️ Network Hub: {viz_target}")
                edges = fetch_visual_network(viz_target)
                if edges:
                    net = build_pyvis_graph(viz_target, edges)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                        net.save_graph(tmp_file.name)
                        tmp_file_path = tmp_file.name
                    with open(tmp_file_path, 'r', encoding='utf-8') as HtmlFile:
                        components.html(HtmlFile.read(), height=500)
                    os.remove(tmp_file_path)
            with col_struct:
                st.markdown(f"#### 🧬 Structural Pockets: {viz_target}")
                uniprot_id = get_uniprot_id(viz_target)
                if uniprot_id:
                    struct_string, struct_format = fetch_alphafold_structure(uniprot_id)
                    if struct_string:
                        residues = get_uniprot_binding_sites(uniprot_id)
                        viewer = render_mutated_protein(struct_string, file_format=struct_format, highlight_residues=residues)
                        showmol(viewer, height=500, width=800)

    st.markdown("---")
    st.subheader("🧪 Bench-to-Cloud Validation Designer")
    if st.session_state.get("ai_targets"):
        b2c_col1, b2c_col2 = st.columns([1, 3])
        with b2c_col1:
            lab_target = st.selectbox("🧬 Select Target for Wet-Lab:", st.session_state.ai_targets, key="lab_select")
            design_btn = st.button("Generate Lab Manifest", type="primary", width="stretch")
        with b2c_col2:
            if design_btn:
                manifest = design_validation_experiment(lab_target, st.session_state.clinical_context.get("cancer_type", ""))
                if manifest:
                    st.success("✅ Experimental Manifest Generated!")
                    st.dataframe(pd.DataFrame(manifest.get('sgrnas', [])), hide_index=True)
                    st.dataframe(pd.DataFrame(manifest.get('primers', [])), hide_index=True)

    st.markdown("---")
    st.subheader("💬 Discuss the Findings")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask a follow up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            chat_sys_msg = f"You are an oncology assistant. Answer based on this report:\n{st.session_state.final_report}"
            msgs = [SystemMessage(content=chat_sys_msg)] + [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            response = ChatOpenAI(model="gpt-5.2", temperature=0.2).invoke(msgs)
            st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

st.sidebar.markdown("---")
st.sidebar.metric(label="Total Tokens Used", value=f"{st.session_state.total_tokens:,}")