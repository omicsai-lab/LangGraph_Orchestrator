import streamlit as st
import requests
import py3Dmol
from stmol import showmol
import re

st.set_page_config(page_title="AlphaFold Mutation Sandbox", layout="wide")

# --- Helper Functions (From our previous sandbox) ---
@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def fetch_alphafold_structure(uniprot_id):
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        api_res = requests.get(api_url)
        if api_res.status_code == 200:
            data = api_res.json()
            if data and isinstance(data, list):
                file_url = data[0].get("pdbUrl")
                file_format = "pdb"
                if not file_url:
                    file_url = data[0].get("cifUrl")
                    file_format = "cif"
                if file_url:
                    struct_res = requests.get(file_url)
                    if struct_res.status_code == 200:
                        return struct_res.text, file_format
        return None, None
    except: return None, None

@st.cache_data(show_spinner=False)
def get_uniprot_binding_sites(uniprot_id):
    """Autonomously fetches known Active Sites and Ligand Binding Pockets from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            features = res.json().get("features", [])
            target_residues = []
            
            # Scan the protein's biology for pockets
            for f in features:
                if f.get("type") in ["Binding site", "Active site"]:
                    loc = f.get("location", {})
                    start = loc.get("start", {}).get("value")
                    end = loc.get("end", {}).get("value")
                    
                    if start and end:
                        # Grab the whole range of the pocket
                        target_residues.extend(list(range(start, end + 1)))
                    elif start:
                        target_residues.append(start)
                        
            # Return unique residues as a list of strings for Py3Dmol
            return [str(r) for r in set(target_residues)]
        return []
    except Exception:
        return []

def extract_residue_number(mutation_string):
    """Extracts the numeric position from a string like 'V600E' or 'G12C'"""
    if not mutation_string:
        return None
    # Find all contiguous digits in the string
    match = re.search(r'\d+', mutation_string)
    if match:
        return match.group()
    return None

def render_mutated_protein(structure_data, file_format="pdb", highlight_residues=None):
    """Renders the protein. Highlights pockets if provided, else falls back to Confidence coloring."""
    view = py3Dmol.view(width=800, height=500)
    view.addModel(structure_data, file_format)
    
    # If we have residues to highlight (Mutation or Active Sites)
    if highlight_residues:
        # Base style: Light grey to make the active sites pop
        view.setStyle({'model': -1}, {"cartoon": {'color': 'lightgrey'}})
        
        # Ensure it's a list even if it's a single mutation
        if isinstance(highlight_residues, str) or isinstance(highlight_residues, int):
            highlight_residues = [str(highlight_residues)]
            
        # Highlight the pocket residues as bright red spheres
        view.addStyle(
            {'resi': highlight_residues}, 
            {'sphere': {'color': 'red', 'radius': 1.2}}
        )
        # Show the chemical structure (sticks) of the pocket
        view.addStyle(
            {'resi': highlight_residues},
            {'stick': {'colorscheme': 'blueCarbon'}}
        )
    else:
        # FALLBACK: If no mutation and no known active site, color by AlphaFold Confidence
        view.setStyle({'model': -1}, {"cartoon": {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
        
    view.zoomTo()
    return view

# --- UI ---
st.title("🎯 Structural Highlighting Sandbox")
st.markdown("Testing programmatic manipulation of 3D coordinates for DNA mutations or Active Sites.")

col1, col2 = st.columns([1, 2])

with col1:
    gene_input = st.text_input("Gene Symbol", value="BRAF")
    
    st.markdown("### Clinical Scenario")
    mode = st.radio("What data do we have?", [
        "DNA Mutation (e.g., BRAF V600E)", 
        "RNA Only (Overexpression - Find Druggable Pocket)"
    ])
    
    residues_to_highlight = None
    
    if "DNA Mutation" in mode:
        mut_input = st.text_input("Mutation (e.g., V600E)", value="V600E")
        residues_to_highlight = extract_residue_number(mut_input)
        
    run_btn = st.button("Render 3D Map", type="primary")

with col2:
    if run_btn and gene_input:
        uniprot_id = get_uniprot_id(gene_input.strip().upper())
        if uniprot_id:
            struct_string, struct_format = fetch_alphafold_structure(uniprot_id)
            if struct_string:
                
                # If RNA only, autonomously hunt for the active site!
                if "RNA Only" in mode:
                    with st.spinner("Hunting UniProt for Active/Binding Sites..."):
                        residues_to_highlight = get_uniprot_binding_sites(uniprot_id)
                        if residues_to_highlight:
                            st.success(f"Autonomously located {len(residues_to_highlight)} Druggable Pocket residues!")
                        else:
                            st.warning("No defined active sites found. Falling back to structural confidence map.")
                else:
                    st.success(f"Highlighting DNA Mutation at position: **{residues_to_highlight}**")
                
                viewer = render_mutated_protein(
                    struct_string, 
                    file_format=struct_format, 
                    highlight_residues=residues_to_highlight
                )
                showmol(viewer, height=500, width=800)
            else:
                st.error("Failed to download structure.")
        else:
            st.error("Invalid Gene Symbol.")