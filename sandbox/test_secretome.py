import streamlit as st
import requests

st.set_page_config(page_title="Secretome Sandbox", layout="wide")

@st.cache_data(show_spinner=False)
def check_biomarker_detectability(gene_symbol: str) -> str:
    """Queries HPA to see if the protein is secreted into blood or exists on the cell surface."""
    # We use 'pc' (Protein class) again, but with strict parsing this time!
    url = f"https://www.proteinatlas.org/api/search_download.php?search={gene_symbol}&format=json&columns=g,pc&compress=no"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            try:
                data = res.json()
            except ValueError:
                return "HPA API returned an invalid response (likely HTML)."
                
            for entry in data:
                if entry.get("Gene", "").upper() == gene_symbol.upper():
                    classes = entry.get("Protein class", [])
                    classes_str = ", ".join(classes) if isinstance(classes, list) else str(classes)
                    
                    # THE STRICT FIX: Look for HPA's exact biological classification tags
                    is_secreted = "Predicted secreted" in classes_str
                    is_membrane = "Predicted membrane" in classes_str
                    
                    if is_secreted:
                        return f"DETECTABILITY: High. {gene_symbol} is a secreted protein, making it a prime candidate for ELISA blood tests or liquid biopsies."
                    elif is_membrane:
                        return f"DETECTABILITY: Moderate. {gene_symbol} is a membrane protein. It may be shed in exosomes or detectable via flow cytometry/CTCs."
                    else:
                        return f"DETECTABILITY: Low. {gene_symbol} is intracellular. Clinical detection would require an invasive tissue biopsy."
            return f"No detectability/secretome data found for {gene_symbol}."
        else:
            return f"Secretome API Error: {res.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

# --- UI ---
st.title("🩸 Blood/Secretome Detectability Sandbox")
st.markdown("Test if a target is secreted into the blood (good for diagnostics) or trapped inside the cell.")

gene_input = st.text_input("Target Gene", value="KLK3") # KLK3 is the official gene name for PSA!

if st.button("Check Detectability", type="primary"):
    with st.spinner(f"Querying HPA Protein Classes for {gene_input}..."):
        result = check_biomarker_detectability(gene_input.strip().upper())
        
        if "High" in result:
            st.success(result)
        elif "Moderate" in result:
            st.warning(result)
        else:
            st.error(result)