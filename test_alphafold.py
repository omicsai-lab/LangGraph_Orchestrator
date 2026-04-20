import streamlit as st
import requests
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="AlphaFold Sandbox", layout="wide")

def get_uniprot_id(hugo_symbol):
    """Maps a HGNC Gene Symbol to a UniProt ID using MyGene.info"""
    url = f"https://mygene.info/v3/query?q=symbol:{hugo_symbol}&fields=uniprot&species=human"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("hits"):
                hit = data["hits"][0]
                uniprot = hit.get("uniprot", {})
                # Try to get the reviewed Swiss-Prot ID first
                if "Swiss-Prot" in uniprot:
                    return uniprot["Swiss-Prot"]
                # Fallback to unreviewed TrEMBL
                elif "TrEMBL" in uniprot:
                    # TrEMBL can be a list or a string
                    return uniprot["TrEMBL"][0] if isinstance(uniprot["TrEMBL"], list) else uniprot["TrEMBL"]
        return None
    except Exception as e:
        st.error(f"MyGene API Error: {e}")
        return None

def fetch_alphafold_structure(uniprot_id):
    """Fetches the 3D coordinates dynamically from the AlphaFold EBI API"""
    # 1. Query the API to get the exact, current file URLs
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        api_res = requests.get(api_url)
        if api_res.status_code == 200:
            data = api_res.json()
            if data and isinstance(data, list):
                # 2. Try to get PDB, but fallback to CIF for massive proteins
                file_url = data[0].get("pdbUrl")
                file_format = "pdb"
                
                if not file_url:
                    file_url = data[0].get("cifUrl")
                    file_format = "cif"
                    
                # 3. Download the actual structure file
                if file_url:
                    struct_res = requests.get(file_url)
                    if struct_res.status_code == 200:
                        return struct_res.text, file_format
        return None, None
    except Exception as e:
        st.error(f"AlphaFold API Error: {e}")
        return None, None

def render_protein(structure_data, file_format="pdb", style="cartoon", color="confidence"):
    """Configures the py3Dmol viewer"""
    view = py3Dmol.view(width=800, height=600)
    
    # Dynamically load as either PDB or CIF
    view.addModel(structure_data, file_format)
    
    # AlphaFold maps confidence (pLDDT) to the b-factor column
    if color == "confidence":
        view.setStyle({'model': -1}, {"cartoon": {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
    else:
        view.setStyle({'model': -1}, {style: {'color': color}})
        
    view.zoomTo()
    return view

# --- UI ---
st.title("🧬 AlphaFold 3D Viewer Sandbox")
st.markdown("Test environment for stmol and py3Dmol integration.")

col1, col2 = st.columns([1, 2])

with col1:
    gene_input = st.text_input("Enter Gene Symbol (e.g., TP53, BRAF, EGFR)", value="BRAF")
    
    style_opts = st.selectbox("Style", ["cartoon", "stick", "sphere"])
    color_opts = st.selectbox("Coloring", ["confidence", "spectrum", "blue", "red"])
    
    run_btn = st.button("Fetch & Render Structure", type="primary")

with col2:
    if run_btn and gene_input:
        with st.spinner(f"Mapping {gene_input} to UniProt ID..."):
            uniprot_id = get_uniprot_id(gene_input.strip().upper())
            
        if not uniprot_id:
            st.error(f"Could not find a valid human UniProt ID for {gene_input}.")
        else:
            st.success(f"Mapped {gene_input} to UniProt ID: **{uniprot_id}**")
            
            with st.spinner("Downloading 3D coordinates from AlphaFold..."):
                struct_string, struct_format = fetch_alphafold_structure(uniprot_id)
                
            if struct_string:
                st.markdown("### Interactive 3D Model")
                st.info("💡 **Tip:** You can click and drag to rotate, and scroll to zoom.")
                
                # Render using our updated function
                viewer = render_protein(struct_string, file_format=struct_format, style=style_opts, color=color_opts)
                
                # Display in Streamlit using stmol
                showmol(viewer, height=600, width=800)
            else:
                st.error("Failed to download structure from AlphaFold DB. (It may not exist in their database).")