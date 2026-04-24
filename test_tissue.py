import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Tissue Artifact Detector", layout="wide")

@st.cache_data(show_spinner=False)
def check_tissue_admixture(gene_symbol):
    """Queries OpenTargets GraphQL for Baseline Tissue Expression"""
    base_url = "https://api.platform.opentargets.org/api/v4/graphql"

    # Step 1: Resolve Hugo Symbol to Ensembl ID
    search_query = """
    query searchTarget($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"]) {
        hits {
          id
          symbol
        }
      }
    }
    """
    try:
        res1 = requests.post(base_url, json={"query": search_query, "variables": {"queryString": gene_symbol}})
        if res1.status_code != 200: 
            return {"status": "Error", "message": "OpenTargets API unreachable."}

        hits = res1.json().get("data", {}).get("search", {}).get("hits", [])
        ensembl_id = next((hit["id"] for hit in hits if hit["symbol"].upper() == gene_symbol.upper()), None)

        if not ensembl_id: 
            return {"status": "Error", "message": "Exact gene match not found in OpenTargets."}

        # Step 2: Fetch Baseline Tissue Expression
        expr_query = """
        query getExpression($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            expressions {
              tissue {
                label
                organs
              }
              rna {
                value
              }
            }
          }
        }
        """
        res2 = requests.post(base_url, json={"query": expr_query, "variables": {"ensemblId": ensembl_id}})
        if res2.status_code != 200: 
            return {"status": "Error", "message": "Failed to fetch expression data."}

        expressions = res2.json().get("data", {}).get("target", {}).get("expressions", [])
        if not expressions: 
            return {"status": "Error", "message": "No expression data available for this gene."}

        # Parse and sort the tissues by highest expression
        expr_data = []
        for exp in expressions:
            rna_val = exp.get("rna", {}).get("value", 0)
            if rna_val > 0:
                tissue = exp.get("tissue", {}).get("label", "Unknown")
                organs = exp.get("tissue", {}).get("organs", [])
                expr_data.append({
                    "Tissue": tissue,
                    "Organ System": ", ".join(organs) if organs else "Unknown",
                    "Expression Level (Baseline)": round(rna_val, 2)
                })

        expr_data = sorted(expr_data, key=lambda x: x["Expression Level (Baseline)"], reverse=True)

        return {
            "status": "Success",
            "gene": gene_symbol,
            "top_tissues": expr_data[:8] # Return top 8 expressing tissues
        }
        
    except Exception as e:
        return {"status": "Error", "message": f"Exception: {str(e)}"}

# --- UI ---
st.title("🔬 Tissue Admixture & Artifact Detector")
st.markdown("Queries OpenTargets (GTEx baseline expression) to determine if your target is tumor-intrinsic or a stromal/immune artifact.")

col1, col2 = st.columns([1, 2])

with col1:
    gene_input = st.text_input("Target Gene", value="LILRB5")
    run_btn = st.button("Check Tissue Specificity", type="primary")

with col2:
    if run_btn and gene_input:
        with st.spinner(f"Querying OpenTargets for {gene_input}..."):
            result = check_tissue_admixture(gene_input.strip().upper())
            
        if result["status"] == "Success":
            st.success("✅ Baseline tissue expression retrieved!")
            
            if result["top_tissues"]:
                st.markdown("#### Top Expressing Tissues")
                df = pd.DataFrame(result["top_tissues"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Dynamic Artifact Warning Engine
                top_organs = str(df["Organ System"].head(3).values).lower()
                top_tissues = str(df["Tissue"].head(3).values).lower()
                
                immune_keywords = ["blood", "bone marrow", "lymph node", "spleen", "macrophage", "t-cell", "b-cell"]
                stromal_keywords = ["adipose", "fat", "fibroblast", "connective", "muscle"]
                
                if any(kw in top_organs or kw in top_tissues for kw in immune_keywords):
                    st.warning("⚠️ **ARTIFACT WARNING:** This gene is predominantly expressed in the immune system (e.g., Blood, Lymph, Marrow). If altered in a solid tumor bulk RNA sample, it likely reflects immune infiltration, NOT a tumor-cell mutation.")
                elif any(kw in top_organs or kw in top_tissues for kw in stromal_keywords):
                    st.warning("⚠️ **ARTIFACT WARNING:** This gene is highly expressed in stromal or adipose tissue. This may represent tissue contamination (e.g., fat cells mixed into a breast biopsy).")
                else:
                    st.info("💡 **Tumor-Intrinsic Likely:** This gene is expressed in solid epithelial/systemic tissues, increasing the likelihood it is a valid target.")
        else:
            st.error(result["message"])