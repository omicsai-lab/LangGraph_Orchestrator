import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Single-Cell Artifact Killer (Gold Standard)", layout="wide")

@st.cache_data(show_spinner=False)
def get_ensembl_id(gene_symbol):
    """Uses HPA's own search API to get its official Ensembl ID, cutting out the middleman!"""
    # We ask HPA specifically for 'g' (Gene) and 'eg' (Ensembl Gene ID)
    url = f"https://www.proteinatlas.org/api/search_download.php?search={gene_symbol}&format=json&columns=g,eg&compress=no"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json()
            if data and isinstance(data, list) and len(data) > 0:
                for entry in data:
                    # Match the gene name exactly
                    if entry.get("Gene", "").upper() == gene_symbol.upper():
                        # Extract the official HPA Ensembl ID
                        ensembl_val = entry.get("Ensembl")
                        # Handle cases where HPA returns a list or a string
                        return ensembl_val[0] if isinstance(ensembl_val, list) else ensembl_val
    except Exception as e:
        print(f"Debug: {e}")
        pass
        
    return None

@st.cache_data(show_spinner=False)
def get_hpa_direct_entry(gene_symbol):
    """Fetches the complete individual entry from HPA using the Ensembl ID"""
    ensembl_id = get_ensembl_id(gene_symbol)
    
    if not ensembl_id:
        return {"status": "Error", "message": f"Could not map {gene_symbol} to an Ensembl ID."}

    url = f"https://www.proteinatlas.org/{ensembl_id}.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json()
            
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
            else:
                entry = data
                
            cell_types = {}
            
            # --- THE UPGRADED HUNTER ---
            for key, val in entry.items():
                # We look for the specific single-cell key, and we expect a DICTIONARY now!
                if "single cell type specific" in key.lower() and isinstance(val, dict):
                    for cell, expr in val.items():
                        try:
                            cell_types[cell.strip()] = float(expr)
                        except ValueError:
                            pass
                    break # Stop hunting once we find it
                            
            if not cell_types:
                return {"status": "Error", "message": "Direct entry retrieved, but no single cell data was found inside.", "raw": entry}
                
            sorted_cells = sorted(cell_types.items(), key=lambda item: item[1], reverse=True)
            
            return {
                "status": "Success",
                "gene": gene_symbol,
                "ensembl_id": ensembl_id,
                "top_cell_types": sorted_cells[:5]
            }
        else:
            return {"status": "Error", "message": f"HPA Direct API failed. Status code: {res.status_code}"}
    except Exception as e:
        return {"status": "Error", "message": f"Exception: {str(e)}"}

# --- UI ---
st.title("🔬 Single-Cell Artifact Killer (Gold Standard)")
st.markdown("Query the complete individual entry on the Human Protein Atlas to definitively map cellular origins.")

col1, col2 = st.columns([1, 2])

with col1:
    gene_input = st.text_input("Target Gene", value="LILRB5")
    run_btn = st.button("Check Cellular Specificity", type="primary")

with col2:
    if run_btn and gene_input:
        with st.spinner(f"Routing through MyGene and fetching HPA master entry for {gene_input}..."):
            result = get_hpa_direct_entry(gene_input.strip().upper())
            
        if result["status"] == "Success":
            st.success(f"✅ Master entry retrieved for {result['ensembl_id']}!")
            
            st.markdown("#### Top Expressing Microscopic Cell Types")
            df = pd.DataFrame(result["top_cell_types"], columns=["Cell Type", "Expression Level (nTPM)"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            top_cell = result["top_cell_types"][0][0].lower()
            immune_words = ["macrophage", "t-cell", "b-cell", "neutrophil", "lymphocyte", "monocyte", "kupffer", "dendritic"]
            stromal_words = ["fibroblast", "adipocyte", "endothelial", "muscle", "stromal"]
            
            if any(kw in top_cell for kw in immune_words):
                st.warning("⚠️ **ARTIFACT WARNING:** This gene is localized to immune cells. In bulk RNA, this reflects TME infiltration, NOT tumor mutations.")
            elif any(kw in top_cell for kw in stromal_words):
                st.warning("⚠️ **ARTIFACT WARNING:** This gene is localized to stromal/structural cells. This likely represents benign tissue contamination.")
            else:
                st.info("💡 **Tumor-Intrinsic Likely:** This gene is expressed in epithelial cells.")
        else:
            st.error(result["message"])
            if "raw" in result:
                with st.expander("🔍 View Raw Master Entry"):
                    st.json(result["raw"])