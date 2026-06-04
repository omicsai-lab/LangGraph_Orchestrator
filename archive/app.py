import streamlit as st
import pandas as pd
import requests
import json
from openai import OpenAI
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==========================================
# PAGE CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="Oncology AI Agent", layout="wide")
st.title("🧬 Precision Oncology Agentic Pipeline")

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    oncokb_key = st.secrets["ONCOKB_API_KEY"]
except KeyError:
    st.error("⚠️ Secrets not found! Please ensure you have a .streamlit/secrets.toml file with your API keys.")
    st.stop()

# ==========================================
# 1. THE AI TOOLS (OncoKB)
# ==========================================
def get_onco_data(hugo, alteration, tumor_type):
    """Fetches real-time drug data from OncoKB using the annotation endpoint."""
    url = "https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange"
    params = {"hugoSymbol": hugo, "alteration": alteration, "tumorType": tumor_type}
    headers = {"accept": "application/json", "Authorization": f"Bearer {oncokb_key}"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            treatments = data.get('treatments', [])
            if not treatments: return {"status": "No drug entries found for this specific alteration."}
            
            results = []
            for treatment in treatments:
                drugs = [d.get('drugName', '') for d in treatment.get('drugs', [])]
                results.append({
                    "drugName": ", ".join(drugs), 
                    "levelOfEvidence": treatment.get('level', 'Unknown')
                })
            return {"status": "Success", "drugs": results}
        return {"status": f"OncoKB API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_onco_data",
            "description": "Query OncoKB for clinical evidence and drug matching for specific gene mutations.",
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
    }
]

# ==========================================
# 2. THE AI ORCHESTRATOR
# ==========================================
def run_drug_search_agent(gene_list, prompt_instructions):
    client = OpenAI(api_key=openai_key)
    
    # We provide a strict template so the AI always includes Level of Evidence
    instructions = f"""
    User Instructions: {prompt_instructions}
    
    Target Genes Identified by PyDESeq2: {json.dumps(gene_list)}
    
    TASK: Use your 'get_onco_data' tool to search for treatments for EACH gene.
    
    CLASSIFICATION RULES:
    1. Known Match: A drug is found, and it is approved for the User's exact Cancer Type.
    2. Novel Target (Repurposing): A drug is found, but it is only approved for a DIFFERENT cancer type.
    3. Novel Target (Un-drugged): The gene is overexpressed, but OncoKB returns zero drug matches.
    
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
          ]
        }}
      ]
    }}
    """
    
    messages = [
        {"role": "system", "content": "You are a specialized Oncology AI Agent. Output valid JSON only."},
        {"role": "user", "content": instructions}
    ]

    with st.status("🧠 AI Agent Investigating Targets...", expanded=True) as status:
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
            sig_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] > 2)].sort_values(by='padj')
            
            agent_payload = []
            for gene in sig_genes.index:
                alt = "V600E" if gene == "BRAF" else "L858R" if gene == "EGFR" else "Amplification"
                agent_payload.append({"hugo": str(gene), "alteration": alt, "tumor_type": disease_interest})
        
        # Run the AI Agent
        final_report_json = run_drug_search_agent(agent_payload, "Find drugs and classify novelty.")
        
        st.markdown("---")
        st.subheader("📊 Clinical Actionability Report")
        
        # --- PARSING THE JSON INTO A BEAUTIFUL UI ---
        try:
            report_data = json.loads(final_report_json)
            
            for target in report_data.get("report", []):
                gene = target.get("gene", "Unknown")
                alt = target.get("alteration", "")
                drugs = target.get("drugs", [])
                
                # Create an expandable section for each gene
                with st.expander(f"🎯 Target: **{gene}** ({alt})", expanded=True):
                    if drugs:
                        # Convert the drugs list into a clean Pandas dataframe for display
                        df_drugs = pd.DataFrame(drugs)
                        # Hide the row index and stretch to full width
                        st.dataframe(df_drugs, hide_index=True, use_container_width=True)
                    else:
                        st.info(f"Novel Target (Un-drugged): No clinical evidence found in OncoKB for {gene} {alt}.")
                        
        except Exception as e:
            st.error("Failed to parse the final report into tables. Showing raw data instead.")
            st.json(final_report_json)
            
    else:
        st.warning("Please upload both the Counts and Metadata CSV files to begin.")