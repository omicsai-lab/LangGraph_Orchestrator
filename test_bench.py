import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="Bench-to-Cloud Sandbox", layout="wide")

# Ensure API key is available
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ OPENAI_API_KEY not found in secrets.")
    st.stop()

# --- 1. DEFINE THE STRICT DATA SCHEMAS ---
class sgRNA(BaseModel):
    target_exon: str = Field(description="Which exon to target (e.g., Exon 2)")
    sequence: str = Field(description="20bp RNA sequence")
    pam: str = Field(description="PAM sequence (e.g., NGG)")
    off_target_risk: str = Field(description="Low, Medium, or High")

class PrimerPair(BaseModel):
    target: str = Field(description="What this primer amplifies (e.g., GAPDH control, or Target Gene)")
    forward: str = Field(description="Forward primer sequence (18-22bp)")
    reverse: str = Field(description="Reverse primer sequence (18-22bp)")
    tm: str = Field(description="Melting temperature (e.g., 60 C)")
    amplicon_size: int = Field(description="Expected amplicon size in bp")

class WetLabManifest(BaseModel):
    rationale: str = Field(description="1-sentence explanation of why we are knocking out this gene.")
    sgrnas: List[sgRNA] = Field(description="Top 2 sgRNA designs for CRISPR knockout.")
    primers: List[PrimerPair] = Field(description="qPCR primers to validate the knockout (include the target gene and 1 housekeeping control).")

# --- 2. THE AI DESIGNER FUNCTION ---
def design_validation_experiment(gene_symbol, cancer_type):
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.1, api_key=openai_key)
    structured_llm = llm.with_structured_output(WetLabManifest)
    
    sys_msg = """You are an expert Molecular Biologist and CRISPR designer.
    The user will provide a target gene that was identified as highly dysregulated in a specific cancer type.
    Your job is to design a realistic, immediate wet-lab validation experiment to knock out this gene using CRISPR-Cas9.
    
    CRITICAL RULES:
    1. Output strictly in the requested JSON format.
    2. Ensure sgRNA sequences are exactly 20 nucleotides + a valid PAM (e.g., CGG, TGG).
    3. Primers should be 18-22 nucleotides with ~50-60% GC content.
    4. Always include a housekeeping gene primer pair (e.g., ACTB or GAPDH) as a control.
    """
    
    prompt = f"Target Gene: {gene_symbol}\nCancer Context: {cancer_type}\nDesign the CRISPR and qPCR validation manifest."
    
    try:
        response = structured_llm.invoke([
            SystemMessage(content=sys_msg), 
            HumanMessage(content=prompt)
        ])
        return response
    except Exception as e:
        st.error(f"AI Design Failed: {e}")
        return None

# --- 3. UI ---
st.title("🧪 Bench-to-Cloud Validation Designer")
st.markdown("Autonomously design CRISPR sgRNAs and qPCR primers for AI-selected targets.")

gene_input = st.text_input("Enter Target Gene", value="BRAF")
cancer_input = st.text_input("Enter Cancer Context", value="Melanoma")

if st.button("Generate Lab Manifest", type="primary"):
    with st.spinner("🤖 AI Molecular Biologist is designing the experiment..."):
        manifest = design_validation_experiment(gene_input, cancer_input)
        
    if manifest:
        st.success("✅ Experimental Manifest Generated!")
        st.markdown(f"**Rationale:** {manifest.rationale}")
        
        st.markdown("### ✂️ CRISPR-Cas9 sgRNA Designs")
        st.warning("⚠️ **Clinical Disclaimer:** These are AI-generated sequences. Verify against the human reference genome using Benchling or IDT before ordering.")
        
        # Format sgRNAs as a clean DataFrame
        sgrna_df = pd.DataFrame([dict(s) for s in manifest.sgrnas])
        st.dataframe(sgrna_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 🧬 qPCR Validation Primers")
        primer_df = pd.DataFrame([dict(p) for p in manifest.primers])
        st.dataframe(primer_df, use_container_width=True, hide_index=True)