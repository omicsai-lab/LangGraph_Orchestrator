import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
from openai import OpenAI
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==========================================
# PAGE CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="Advanced Oncology AI Agent", layout="wide")
st.title("🧬 Advanced Precision Oncology Pipeline")

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    oncokb_key = st.secrets["ONCOKB_API_KEY"]
except KeyError:
    st.error("⚠️ Secrets not found! Please ensure you have a .streamlit/secrets.toml file with your API keys.")
    st.stop()

# ==========================================
# 1. THE AI TOOLS (OncoKB & PubMed)
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
                # Extract PMIDs directly from OncoKB
                pmids = treatment.get('pmids', []) 
                results.append({
                    "drugName": ", ".join(drugs), 
                    "levelOfEvidence": treatment.get('level', 'Unknown'),
                    "pmids": pmids
                })
            return {"status": "Success", "drugs": results}
        return {"status": f"OncoKB API Error: {response.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def search_pubmed(gene, tumor_type):
    """Searches the free NCBI PubMed API for experimental literature."""
    query = f"{gene}[Gene] AND {tumor_type} AND targeted therapy"
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 3 # Bring back the top 3 papers
    }
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list: return {"status": "No experimental literature found."}
            return {"status": "Success", "experimental_pmids": id_list}
        return {"status": f"PubMed Error: {res.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_onco_data",
            "description": "Query OncoKB for clinical evidence and drugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hugo": {"type": "string"},
                    "alteration": {"type": "string"},
                    "tumor_type": {"type": "string"}
                },
                "required": ["hugo", "alteration", "tumor_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "Query PubMed for experimental literature on novel or un-drugged targets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "tumor_type": {"type": "string"}
                },
                "required": ["gene", "tumor_type"]
            }
        }
    }
]

# ==========================================
# 2. THE AI ORCHESTRATOR
# ==========================================
def run_drug_search_agent(gene_list, prompt_instructions):
    client = OpenAI(api_key=openai_key)
    
    instructions = f"""
    User Instructions: {prompt_instructions}
    Target Genes Identified: {json.dumps(gene_list)}
    
    TASK: Use 'get_onco_data' to find drugs. If a target has NO drugs, or is a repurposed novel target, use 'search_pubmed' to find experimental papers.
    
    CRITICAL: Output a structured JSON report exactly matching this schema:
    {{
      "report": [
        {{
          "gene": "BRAF",
          "alteration": "V600E",
          "tumor_type": "Melanoma",
          "drugs": [
            {{
              "Drug Name": "Dabrafenib",
              "Level of Evidence": "LEVEL_1",
              "Classification": "Known Match"
            }}
          ],
          "literature": [
            {{
              "PMID": "223985",
              "Source": "OncoKB Evidence" 
            }},
            {{
              "PMID": "998877",
              "Source": "Experimental Search"
            }}
          ]
        }}
      ]
    }}
    """
    
    messages = [
        {"role": "system", "content": "You are a specialized Oncology AI Agent. Output valid JSON only."},
        {"role": "user", "content": instructions}
    ]

    with st.status("🧠 AI Agent Investigating Targets & Literature...", expanded=True) as status:
        while True:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                response_format={"type": "json_object"}
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)

            if not response_message.tool_calls:
                status.update(label="AI Investigation Complete!", state="complete", expanded=False)
                break 

            for tool_call in response_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "get_onco_data":
                    st.write(f"🔍 Searching OncoKB for **{args['hugo']}**...")
                    result = get_onco_data(args['hugo'], args['alteration'], args['tumor_type'])
                elif tool_call.function.name == "search_pubmed":
                    st.write(f"📚 Searching PubMed for experimental data on **{args['gene']}**...")
                    result = search_pubmed(args['gene'], args['tumor_type'])
                    
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": json.dumps(result)
                })
                
    return response_message.content

# ==========================================
# 3. MAIN UI & DESEQ2 LOGIC
# ==========================================
st.subheader("1. Upload Cohort Data")
col1, col2 = st.columns(2)
with col1:
    counts_file = st.file_uploader("Upload RNA-seq Counts (CSV)", type=["csv"])
with col2:
    metadata_file = st.file_uploader("Upload Metadata (CSV)", type=["csv"])

disease_interest = st.text_input("Cancer Type / Disease Interest", value="Melanoma")

if st.button("Run Pipeline (PyDESeq2 + AI Agent)", type="primary"):
    if counts_file and metadata_file:
        counts_df = pd.read_csv(counts_file, index_col=0)
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        
        with st.spinner("🧬 Running Differential Expression Analysis (Tumor vs Normal)..."):
            dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors="condition")
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"])
            stat_res.summary()
            
            results_df = stat_res.results_df
            
            # --- VOLCANO PLOT ---
            st.markdown("---")
            st.subheader("📈 Gene Expression Volcano Plot")
            plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
            plot_df['-log10(padj)'] = -np.log10(plot_df['padj'] + 1e-300)
            conditions = [(plot_df['padj'] < 0.05) & (plot_df['log2FoldChange'] > 2), (plot_df['padj'] < 0.05) & (plot_df['log2FoldChange'] < -2)]
            choices = ['Upregulated (Target)', 'Downregulated']
            plot_df['Significance'] = np.select(conditions, choices, default='Not Significant')
            fig = px.scatter(plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', color_discrete_map={'Upregulated (Target)': 'red', 'Downregulated': 'blue', 'Not Significant': 'lightgrey'}, hover_name=plot_df.index, title="Tumor vs. Normal Sample Expression")
            st.plotly_chart(fig, use_container_width=True)

            # Filter for the agent
            sig_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] > 2)].sort_values(by='padj')
            agent_payload = [{"hugo": str(gene), "alteration": "V600E" if gene == "BRAF" else "L858R" if gene == "EGFR" else "Amplification", "tumor_type": disease_interest} for gene in sig_genes.index]
        
        # Run the AI Agent
        final_report_json = run_drug_search_agent(agent_payload, "Find drugs, classify novelty, and gather literature.")
        
        st.markdown("---")
        st.subheader("📊 Clinical Actionability Report")
        
        try:
            report_data = json.loads(final_report_json)
            for target in report_data.get("report", []):
                gene = target.get("gene", "Unknown")
                alt = target.get("alteration", "")
                drugs = target.get("drugs", [])
                literature = target.get("literature", [])
                
                with st.expander(f"🎯 Target: **{gene}** ({alt})", expanded=True):
                    
                    # 1. Display Drugs
                    st.markdown("##### 💊 Potential Therapeutics")
                    if drugs:
                        st.dataframe(pd.DataFrame(drugs), hide_index=True, use_container_width=True)
                    else:
                        st.info(f"Novel Target (Un-drugged): No clinical evidence found in OncoKB for {gene}.")
                    
                    # 2. Display Literature Sources
                    st.markdown("##### 📚 Supporting Literature")
                    if literature:
                        for lit in literature:
                            pmid = lit.get("PMID", "N/A")
                            source = lit.get("Source", "Unknown")
                            # Make the PMID a clickable link to PubMed!
                            st.markdown(f"- **{source}**: [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                    else:
                        st.write("No specific literature found.")
                        
        except Exception as e:
            st.error("Failed to parse the final report into tables. Showing raw data instead.")
            st.json(final_report_json)
            
    else:
        st.warning("Please upload both the Counts and Metadata CSV files to begin.")