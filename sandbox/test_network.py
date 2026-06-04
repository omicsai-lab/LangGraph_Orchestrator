import streamlit as st
import requests
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

st.set_page_config(page_title="Network Sandbox", layout="wide")

@st.cache_data(show_spinner=False)
def get_protein_network(hugo_symbol, max_nodes=10):
    """Fetches the interacting network from STRING DB"""
    url = f"https://string-db.org/api/json/network?identifiers={hugo_symbol}&species=9606&limit={max_nodes}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
        return []
    except:
        return []

def build_pyvis_graph(central_gene, edges_data):
    """Builds a physics-based interactive graph"""
    # Initialize NetworkX graph
    G = nx.Graph()
    
    # Add nodes and edges
    for edge in edges_data:
        node_a = edge.get("preferredName_A")
        node_b = edge.get("preferredName_B")
        score = edge.get("score", 0) # Confidence score
        
        if node_a and node_b:
            # Only add edges with decent confidence (>0.4)
            if score > 0.4:
                G.add_edge(node_a, node_b, weight=score)

    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="white")
    
    # Force the central gene to be a red star, others as blue dots
    for node in G.nodes():
        if node == central_gene:
            net.add_node(node, label=node, color="#EF553B", size=30, shape="star")
        else:
            # Degree centrality controls size of other nodes
            size = 15 + (G.degree(node) * 2)
            net.add_node(node, label=node, color="#636EFA", size=size)

    # Add the edges
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]['weight'], color="#4A4A4A")

    # Turn on physics for that cool bouncy effect
    net.toggle_physics(True)
    return net

# --- UI ---
st.title("🕸️ Interactive Systems Biology Sandbox")
st.markdown("Test environment for PyVis and NetworkX.")

central_gene = st.text_input("Enter Central Gene Hub (e.g., EGFR, TP53, MYC)", value="EGFR")
run_btn = st.button("Generate Network", type="primary")

if run_btn and central_gene:
    with st.spinner("Fetching STRING API interactions..."):
        # We fetch a slightly larger network (15 nodes) for a better visual web
        edges = get_protein_network(central_gene.strip().upper(), max_nodes=15)
        
    if edges:
        st.success(f"Network built for {central_gene}")
        
        # Build the graph
        net = build_pyvis_graph(central_gene.strip().upper(), edges)
        
        # Save to a temporary HTML file and render in Streamlit
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_file_path = tmp_file.name # Save the path string
            
        # Safely open, read, and auto-close the file using a context manager
        with open(tmp_file_path, 'r', encoding='utf-8') as HtmlFile:
            source_code = HtmlFile.read()
            components.html(source_code, height=650)
            
        # Now that the file is fully closed, Windows will let us delete it!
        os.remove(tmp_file_path)
    else:
        st.error("Could not find network data for this gene.")