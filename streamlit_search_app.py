# streamlit_search_app.py
# Solid, defensive Streamlit search app that:
#  - combines CSVs from a folder
#  - detects/builds a text column
#  - creates & caches embeddings
#  - performs semantic search and shows results
#
# Usage:
# 1) put your CSV files in a folder (default: ./Combined_data)
# 2) run: streamlit run streamlit_search_app.py

import pandas as pd
import numpy as np
import os
import glob
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Folder where your CSV files are stored
folder_path = r"C:\Users\FireA\Documents\GitHub\SciSearch\Combined_data"

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Load each CSV and combine
dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except:
        st.warning(f"Could not read file: {f}")

if not dfs:
    st.error("No CSV files could be read. Check your folder and CSVs!")
    st.stop()

combined_df = pd.concat(dfs, ignore_index=True, sort=False)
combined_df = combined_df.reset_index(drop=True)

st.write(f"✅ Combined dataset shape: {combined_df.shape}")
st.write("Columns detected:", combined_df.columns.tolist())

st.set_page_config(page_title="NASA BioSearch", layout="wide")
# Make sure we use the correct column for semantic search
TEXT_COLUMN = "Abstract"  # matches your CSV

# Drop old embeddings if they exist
if 'embedding' in combined_df.columns:
    combined_df = combined_df.drop(columns=['embedding'])

# Ensure text column is string type
combined_df[TEXT_COLUMN] = combined_df[TEXT_COLUMN].fillna('').astype(str)
# Load embedding model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
@st.cache_data(show_spinner=False)
def compute_embeddings(texts_tuple):
    # texts_tuple must be a tuple for caching
    texts = list(texts_tuple)
    arr = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return arr.tolist()

# Compute embeddings
with st.spinner("Generating embeddings..."):
    texts_tuple = tuple(combined_df[TEXT_COLUMN].tolist())
    combined_df['embedding'] = compute_embeddings(texts_tuple)
def semantic_search(query, df, top_k=10):
    if not query.strip():
        return pd.DataFrame()  # empty if no query
    qvec = model.encode([query], convert_to_numpy=True)[0]
    emb_array = np.array(df['embedding'].tolist())
    sims = cosine_similarity([qvec], emb_array)[0]
    df2 = df.copy()
    df2['similarity'] = sims
    return df2.sort_values(by='similarity', ascending=False).head(top_k)
st.subheader("Search NASA Publications")
query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if query:
    results = semantic_search(query, combined_df, top_k)
    if results.empty:
        st.info("No results found. Try a different query.")
    else:
        for i, row in results.iterrows():
            st.markdown(f"### {row['Title']}")
            st.markdown(f"**Authors:** {row['Author Names']}")
            st.markdown(f"{row['Abstract'][:500]}...")
            st.markdown(f"**Similarity score:** {row['similarity']:.3f}")
            st.markdown("---")


# ------------------------------
# Helper functions
# ------------------------------
def list_csv_files(folder):
    pattern = os.path.join(folder, "*.csv")
    return sorted(glob.glob(pattern))

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        # Try with low_memory=False as fallback
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as e2:
            return None

def find_first_col(df, candidates):
    # Return actual column name (case-sensitive) if any candidate matches ignoring case
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def detect_or_build_text_column(df):
    # 1) look for common text columns
    text_candidates = ['abstract', 'summary', 'description', 'text', 'content', 'body', 'paper_abstract']
    chosen = find_first_col(df, text_candidates)
    if chosen:
        return chosen, False  # found, not created

    # 2) attempt to build a combined text column from other likely fields
    parts_priority = ['title', 'summary', 'description', 'results', 'conclusion', 'notes', 'keywords', 'abstract']
    found_parts = []
    lc = {c.lower(): c for c in df.columns}
    for p in parts_priority:
        if p.lower() in lc:
            if lc[p.lower()] not in found_parts:
                found_parts.append(lc[p.lower()])

    if not found_parts:
        # nothing useful to combine
        return None, False

    # create combined_text column
    combined_series = df[found_parts].fillna('').astype(str).agg(' '.join, axis=1)
    out_col = 'combined_text'
    # ensure we don't overwrite
    i = 0
    while out_col in df.columns:
        i += 1
        out_col = f'combined_text_{i}'
    df[out_col] = combined_series
    return out_col, True

# ------------------------------
# UI: choose folder & load CSVs
# ------------------------------
st.title("🔎 NASA BioScience — Semantic Search (solid)")

default_folder = os.path.join(os.getcwd(), "Combined_data")
folder_path = st.text_input("Folder containing your CSV files (one folder with all CSVs)", value=default_folder)

if not os.path.isdir(folder_path):
    st.error(f"Folder not found: {folder_path}\nCreate the folder and put your CSV files there, or paste the correct path above.")
    st.stop()

csv_files = list_csv_files(folder_path)
if not csv_files:
    st.error(f"No CSV files found in folder: {folder_path}\nPut your CSV files there (extension .csv).")
    st.stop()

st.sidebar.info(f"Found {len(csv_files)} CSV file(s).")
if st.sidebar.button("Show files"):
    st.sidebar.write(csv_files)

# Combine CSVs (robustly)
dfs = []
bad_files = []
for f in csv_files:
    df_ = safe_read_csv(f)
    if df_ is None:
        bad_files.append(f)
    else:
        dfs.append(df_)

if bad_files:
    st.warning("Some files could not be read. They were skipped. Files:\n" + "\n".join(bad_files))

if not dfs:
    st.error("No CSV files were successfully read. Fix the files and try again.")
    st.stop()

combined_df = pd.concat(dfs, ignore_index=True, sort=False)
combined_df = combined_df.reset_index(drop=True)

st.write("✅ Combined dataset shape:", combined_df.shape)
st.write("Columns detected:", combined_df.columns.tolist())

# ------------------------------
# Detect text / title / author / date columns
# ------------------------------
text_col, created_flag = detect_or_build_text_column(combined_df)
if text_col is None:
    st.error("Could not find a text column (abstract/summary/description) and couldn't build one automatically. "
             "Columns found: " + ", ".join(combined_df.columns.tolist()) + 
             "\n\nIf your dataset has the abstracts in a column with an unusual name, update the CSV header or rename the column.")
    st.stop()

title_col = find_first_col(combined_df, ['title', 'paper_title', 'headline', 'name'])
authors_col = find_first_col(combined_df, ['authors', 'author', 'creators'])
date_col = find_first_col(combined_df, ['date', 'year', 'publication_date', 'pub_date'])

st.sidebar.markdown("**Auto-detected columns**")
st.sidebar.write({
    "text_column": text_col,
    "title_column": title_col or "(none)",
    "authors_column": authors_col or "(none)",
    "date_column": date_col or "(none)",
})

# Offer the user option to override detected text column
st.info(f"Using **{text_col}** as the text column (created: {created_flag}). You can override below.")
override_text = st.text_input("Override text column name (leave empty to use detected)", value="")
if override_text.strip():
    if override_text in combined_df.columns:
        text_col = override_text
        st.success(f"Using overridden text column: {text_col}")
    else:
        st.warning(f"'{override_text}' not found in columns; keeping detected column '{text_col}'.")
# Drop old embeddings if text column changed
if 'embedding' in combined_df.columns:
    combined_df = combined_df.drop(columns=['embedding'])

# ensure text column is string
combined_df[text_col] = combined_df[text_col].fillna('').astype(str)

# ------------------------------
# Load embedding model (safe)
# ------------------------------
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    st.error("The package 'sentence-transformers' is required but not installed.\nRun:\n\npip install sentence-transformers\n\nthen restart the app.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    # small, fast model good for search demos
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Loading embedding model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

# ------------------------------
# Compute or reuse embeddings
# ------------------------------
@st.cache_data(show_spinner=False)
def compute_embeddings(texts_tuple):
    # texts_tuple must be a tuple (hashable) for caching
    texts = list(texts_tuple)
    arr = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # convert to Python lists so caching/serialization is safe
    return arr.tolist()

# If embeddings exist but columns may be stale, we rebuild only if missing or length mismatch
need_embeddings = ('embedding' not in combined_df.columns) or (len(combined_df) != len(combined_df['embedding']) if 'embedding' in combined_df.columns else True)

if need_embeddings:
    st.info("Generating embeddings for the publications (cached). This will take some time on first run.")
    texts_tuple = tuple(combined_df[text_col].tolist())
    try:
        embeddings_list = compute_embeddings(texts_tuple)
        combined_df['embedding'] = embeddings_list
        # optionally save combined csv with embeddings omitted (embeddings are heavy)
        combined_df.to_csv(os.path.join(folder_path, "combined_df.csv"), index=False)
    except Exception as e:
        st.error(f"Error while creating embeddings: {e}")
        st.stop()
else:
    st.success("Embeddings already present and kept.")

# ------------------------------
# Prepare columns used for display
# ------------------------------
def safe_col_name(df, candidates):
    c = find_first_col(df, candidates)
    return c

display_title = safe_col_name(combined_df, ['title','paper_title','headline','name']) or "(no title column)"
display_authors = safe_col_name(combined_df, ['authors','author','creators'])  # may be None
display_date = safe_col_name(combined_df, ['date','year','publication_date','pub_date'])  # may be None

# ------------------------------
# UI: Search box + filters
# ------------------------------
st.markdown("---")
st.subheader("Search publications")

col1, col2, col3 = st.columns([4,1,1])

with col1:
    query = st.text_input("Enter search query (semantic / natural language):", value="")
with col2:
    top_k = st.slider("Results", min_value=1, max_value=50, value=10)
with col3:
    if display_authors and display_authors in combined_df.columns:
        authors_unique = combined_df[display_authors].dropna().unique().tolist()
        authors_filter = st.multiselect("Filter authors", options=sorted([str(a) for a in authors_unique]), help="Optional")
    else:
        authors_filter = []

# Apply author filter
df_filtered = combined_df
if authors_filter and display_authors:
    df_filtered = combined_df[combined_df[display_authors].astype(str).isin(authors_filter)]

st.write(f"Filtered dataset: {len(df_filtered)} rows")

# ------------------------------
# Search logic
# ------------------------------
def semantic_search(query, df, top_k=5):
    if not query or str(query).strip()=="":
        return pd.DataFrame()  # empty
    try:
        qvec = model.encode([query], convert_to_numpy=True)[0]
    except Exception as e:
        st.error(f"Model failed to encode the query: {e}")
        return pd.DataFrame()

    # build numpy array of embeddings
    try:
        emb_array = np.array(df['embedding'].tolist())
    except Exception as e:
        st.error(f"Could not form embeddings array: {e}")
        return pd.DataFrame()

    if emb_array.ndim != 2:
        st.error("Embeddings have unexpected shape.")
        return pd.DataFrame()

    sims = cosine_similarity([qvec], emb_array)[0]
    df2 = df.copy()
    df2['similarity'] = sims
    df2_sorted = df2.sort_values(by='similarity', ascending=False).head(top_k)
    return df2_sorted

# ------------------------------
# Display results
# ------------------------------
if query:
    with st.spinner("Searching..."):
        results = semantic_search(query, df_filtered, top_k=top_k)
    if results is None or results.empty:
        st.info("No results (try a different query).")
    else:
        st.success(f"Showing top {len(results)} results for: '{query}'")
        # Styling
        card_css = """
        <style>
        .card { padding:16px; margin-bottom:12px; background: #ffffff; border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);}
        .small { font-size: 0.85em; color: #444; }
        </style>
        """
        st.markdown(card_css, unsafe_allow_html=True)

        for _, row in results.iterrows():
            title = row.get(display_title, "") if display_title in row.index else row.get(text_col, "")[:80]
            authors = row.get(display_authors, "") if display_authors and display_authors in row.index else ""
            datev = row.get(display_date, "") if display_date and display_date in row.index else ""
            txt = row.get(text_col, "")
            sim = row.get('similarity', 0.0)

            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex; align-items:center; justify-content:space-between">
                    <div style="flex:1">
                      <h3 style="margin:2px 0">{title}</h3>
                      <div class="small"><b>Authors:</b> {authors} &nbsp; <b>Date:</b> {datev} &nbsp; <b>Score:</b> {sim:.3f}</div>
                    </div>
                  </div>
                  <div style="margin-top:8px; color:#111">{(txt[:1000] + '...') if len(txt) > 1000 else txt}</div>
                </div>
                """, unsafe_allow_html=True)

        # allow download of results
        csv_bytes = results.drop(columns=['embedding']).to_csv(index=False).encode('utf-8')
        st.download_button("Download results (CSV)", data=csv_bytes, file_name="search_results.csv", mime="text/csv")

# ------------------------------
# Footer: quick tips
# ------------------------------
st.markdown("---")
st.info("Notes: This app automatically combines CSV files in the folder and attempts to detect or build a text column. "
        "If your dataset uses unusual column names, rename them or override the text column above.")
