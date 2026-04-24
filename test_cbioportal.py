import streamlit as st
import requests

st.set_page_config(page_title="TCGA Feasibility Sandbox", layout="wide")

@st.cache_data(show_spinner=False)
def get_entrez_id(hugo_symbol):
    """Maps HGNC symbol to Entrez ID using MyGene.info"""
    url = f"https://mygene.info/v3/query?q=symbol:{hugo_symbol}&fields=entrezgene&species=human"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("hits"):
                return data["hits"][0].get("entrezgene")
        return None
    except Exception as e:
        st.error(f"MyGene API Error: {e}")
        return None

@st.cache_data(show_spinner=False)
def check_tcga_feasibility(hugo_symbol, study_id="brca_tcga_pan_can_atlas_2018"):
    """Queries cBioPortal API for alteration frequencies in a specific TCGA cohort."""
    entrez_id = get_entrez_id(hugo_symbol)
    if not entrez_id:
        return {"status": "Error", "message": f"Could not map {hugo_symbol} to an Entrez ID."}

    base_url = "https://www.cbioportal.org/api"
    
    try:
        # 1. Get total number of patients/samples in the study
        samples_url = f"{base_url}/studies/{study_id}/samples"
        samples_res = requests.get(samples_url)
        if samples_res.status_code != 200:
            return {"status": "Error", "message": "Failed to fetch study cohort size."}
        total_samples = len(samples_res.json())

        # Define the default "all samples" list ID for the study
        sample_list_id = f"{study_id}_all"

        # 2. Check for Mutations
        mut_profile = f"{study_id}_mutations"
        # NEW: Added &sampleListId to the URL!
        mut_url = f"{base_url}/molecular-profiles/{mut_profile}/mutations?entrezGeneId={entrez_id}&sampleListId={sample_list_id}"
        mut_res = requests.get(mut_url)
        mutated_samples = set()
        if mut_res.status_code == 200:
            for m in mut_res.json():
                mutated_samples.add(m.get("sampleId"))
        else:
            print(f"Mutation API Error: {mut_res.text}") # For background debugging

        # 3. Check for Copy Number Alterations (Amplifications/Deletions)
        cna_profile = f"{study_id}_cna"
        # NEW: Added &sampleListId to the URL!
        cna_url = f"{base_url}/molecular-profiles/{cna_profile}/discrete-copy-number-alterations?entrezGeneId={entrez_id}&sampleListId={sample_list_id}"
        cna_res = requests.get(cna_url)
        altered_cna_samples = set()
        if cna_res.status_code == 200:
            for c in cna_res.json():
                # cBioPortal CNA values: 2 = Amplification, -2 = Deep Deletion
                if c.get("alteration") in [2, -2]: 
                    altered_cna_samples.add(c.get("sampleId"))

        # 4. Calculate total unique altered patients
        total_altered = len(mutated_samples.union(altered_cna_samples))
        alteration_rate = (total_altered / total_samples) * 100 if total_samples > 0 else 0

        return {
            "status": "Success",
            "study_id": study_id,
            "total_samples": total_samples,
            "altered_samples": total_altered,
            "mutation_count": len(mutated_samples),
            "cna_count": len(altered_cna_samples),
            "alteration_rate_percent": round(alteration_rate, 2)
        }

    except Exception as e:
        return {"status": "Error", "message": f"cBioPortal API Error: {str(e)}"}

# --- UI ---
st.title("📊 TCGA Population Reality Check")
st.markdown("Query the cBioPortal REST API to see if your target is actually dysregulated in real-world human cohorts.")

col1, col2 = st.columns([1, 2])

with col1:
    gene_input = st.text_input("Target Gene", value="TP53")
    
    # Hardcoded a few common TCGA studies for the sandbox
    tcga_studies = {
        "Breast Cancer (BRCA)": "brca_tcga_pan_can_atlas_2018",
        "Melanoma (SKCM)": "skcm_tcga_pan_can_atlas_2018",
        "Lung Adenocarcinoma (LUAD)": "luad_tcga_pan_can_atlas_2018",
        "Glioblastoma (GBM)": "gbm_tcga_pan_can_atlas_2018"
    }
    cancer_input = st.selectbox("TCGA Cohort", list(tcga_studies.keys()))
    
    run_btn = st.button("Check Feasibility", type="primary")

with col2:
    if run_btn and gene_input:
        study_id = tcga_studies[cancer_input]
        with st.spinner(f"Querying cBioPortal for {gene_input} in {study_id}..."):
            result = check_tcga_feasibility(gene_input.strip().upper(), study_id)
            
        if result["status"] == "Success":
            st.success("✅ Population data retrieved!")
            
            rate = result["alteration_rate_percent"]
            
            # Dynamic Insight Generation
            if rate > 10:
                st.info(f"💡 **High Feasibility:** {gene_input} is altered in **{rate}%** of {cancer_input} patients. This is a highly relevant clinical target.")
            elif rate > 2:
                st.warning(f"⚠️ **Moderate Feasibility:** {gene_input} is altered in **{rate}%** of patients. This represents a niche sub-population.")
            else:
                st.error(f"🚨 **Low Feasibility:** {gene_input} is altered in only **{rate}%** of patients. Targeting this may result in an underpowered clinical study.")
                
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Cohort Size", result["total_samples"])
            col_b.metric("Altered Patients", result["altered_samples"])
            col_c.metric("Alteration Rate", f"{rate}%")
            
            st.write(f"*(Breakdown: {result['mutation_count']} mutations, {result['cna_count']} deep amplifications/deletions)*")
        else:
            st.error(result["message"])