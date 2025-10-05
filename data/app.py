import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer

# Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Setup nltk ---
def ensure_nltk_resource(resource_name):
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
    except LookupError:
        nltk.download(resource_name)

ensure_nltk_resource('punkt')

# --- Summarizer ---
summarizer = LsaSummarizer()

def summarize(text, sentences_count=3):
    if not text or text.strip() == "":
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# --- Fetch PubMed ---
def fetch_pubmed(query, max_results=10):
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    resp = requests.get(esearch_url, params=params)
    pmids = resp.json().get('esearchresult', {}).get('idlist', [])
    if not pmids:
        return []
    fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    data = requests.get(efetch_url, params=fetch_params).text
    root = ET.fromstring(data)
    records = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="N/A")
        abstract = article.findtext(".//AbstractText", default="")
        authors = ", ".join([a.findtext("LastName", "") for a in article.findall(".//Author") if a.find("LastName") is not None])
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
        records.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": authors,
            "Source": "PubMed",
            "URL": url
        })
    return records

# --- Fetch NASA ADS ---
def fetch_nasa_ads(query, max_results=10, token=None):
    if token is None:
        return []
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "fl": "title,author,abstract,pubdate,bibcode,doctype",
        "rows": max_results,
        "format": "json"
    }
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        st.error(f"NASA ADS API error: {resp.status_code} - {resp.text}")
        return []
    data = resp.json()
    records = []
    for doc in data.get("response", {}).get("docs", []):
        bibcode = doc.get("bibcode", "")
        url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
        records.append({
            "Title": doc.get("title", [""])[0],
            "Authors": ", ".join(doc.get("author", [])),
            "Abstract": doc.get("abstract", ""),
            "PubDate": doc.get("pubdate", ""),
            "Source": "NASA ADS",
            "URL": url
        })
    return records

# --- Load FLAN-T5 locally ---
@st.cache_resource
def load_flant5(model_name="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_flant5("google/flan-t5-large")


def generate_ai_answer(question, docs, max_docs=5):
    context = "\n\n".join([doc.get("Abstract", "") for doc in docs[:max_docs] if doc.get("Abstract")])
    if not context:
        return "No abstracts available to answer the question."

    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

<<<<<<< HEAD
    # define model + payload
    model_id = "google/flan-t5-large"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}

    # make API call
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers=headers,
            json=payload,
        )
        res.raise_for_status()
        data = res.json()

        # parse result
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        else:
            return f"no answer generated — raw response: {data}"
    except Exception as e:
        import traceback
        return f"AI generation failed:\n{traceback.format_exc()}"




# -------------------
# Streamlit UI
# -------------------
[theme]
primarycolor = "#4CAF50"
base = "light" or "dark"
baseFontSize = 16
baseFontWeight = "400"
chartColor = "#FF4B4B"
sidebarBorder = true
font = "sans-serif"
headingFont = "sans-serif"
codeFont = "monospace"
headingFontWeight = "700"
codeFontWeight = "400"
linkColor = "#1a73e8"
textColor = "#262730"
linkColor = "#1a73e8"
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
widgetBackgroundColor = "#F0F2F6"
codeBackgroundColor = "#F6F6F6"
dataframeHeaderBackgroundColor = "#F0F2F6"
borderRadius = 8
borderColor = "#E6E6E6"
sidebarBorderColor = "#E6E6E6"
dataframeBorderColor = "#E6E6E6"
paletteRed = "#FF4B4B"
paletteOrange = "#FFA500"
paletteYellow = "#FFD700"
paletteGreen = "#21BA45"
paletteBlue = "#1A73E8"
paletteViolet = "#A259FF"
 paletteGray = "#808080"
=======
# --- Streamlit UI ---
>>>>>>> 26e4bfac0694010bbeeace2941fd81fbb377a6c0
st.set_page_config(page_title="SciSearch", layout="wide")
st.title("SciSearch: PubMed & NASA ADS Explorer")

<<<<<<< HEAD
# Dark/light toggle
mode = st.sidebar.radio("Theme Mode:", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("<style>body{background-color:#222;color:#eee}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color:#fff;color:#000}</style>", unsafe_allow_html=True)

# Signup form
with st.sidebar.form("signup_form"):
    st.header("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Sign Up")
    if submitted:
        st.success(f"Welcome, {username}!")
        st.session_state["user"] = username

# Search inputs
query = st.sidebar.text_input("Search term:", value="microgravity muscle atrophy")
source_filter = st.sidebar.multiselect("Sources:", ["PubMed", "NASA ADS"], default=["PubMed", "NASA ADS"])
max_results = st.sidebar.slider("Max results per source:", 5, 30, 15)
fetch_button = st.sidebar.button("Fetch Studies")

# session_state caching
if "all_results" not in st.session_state:
    st.session_state["all_results"] = []

if fetch_button:
    all_results = []
    if "PubMed" in source_filter:
        with st.spinner("Fetching PubMed studies..."):
            all_results.extend(fetch_pubmed(query, max_results))
    if "NASA ADS" in source_filter:
        nasa_token = st.secrets.get("NASA_ADS_API_TOKEN")
        with st.spinner("Fetching NASA ADS studies..."):
            all_results.extend(fetch_nasa_ads(query, max_results, nasa_token))
    for doc in all_results:
        doc["Summary"] = summarize(doc["Abstract"])
    st.session_state["all_results"] = all_results

# Tabs
tab1, tab2 = st.tabs(["Studies Explorer", "AI Q&A"])
=======
tab1, tab2 = st.tabs(["Studies Explorer", "AI Overview"])
>>>>>>> 26e4bfac0694010bbeeace2941fd81fbb377a6c0

with tab1:
    st.header("Search & Explore Studies")
    query = st.text_input("Enter search term:", value="microgravity muscle atrophy")
    source_filter = st.multiselect("Filter by source", ["PubMed", "NASA ADS"], default=["PubMed"])
    max_results = st.slider("Max results per source", 5, 20, 10)

    if st.button("Search Studies"):
        all_results = []
        if "PubMed" in source_filter:
            with st.spinner("Fetching PubMed studies..."):
                all_results.extend(fetch_pubmed(query, max_results))
        if "NASA ADS" in source_filter:
            with st.spinner("Fetching NASA ADS studies..."):
                nasa_token = st.secrets.get("NASA_ADS_API_TOKEN")
                all_results.extend(fetch_nasa_ads(query, max_results, nasa_token))

        for doc in all_results:
            doc["Summary"] = summarize(doc.get("Abstract", ""))

        if all_results:
            st.success(f"Found {len(all_results)} studies.")
            df = pd.DataFrame(all_results)
            for idx, row in df.iterrows():
                st.subheader(row["Title"])
                st.markdown(f"**Authors:** {row['Authors']}")
                st.markdown(f"**Source:** {row['Source']}")
                if "PubDate" in row and row["PubDate"]:
                    st.markdown(f"**Publication Date:** {row['PubDate']}")
                st.markdown(f"**Summary:** {row['Summary']}")
                st.markdown(f"[Open Study]({row['URL']})", unsafe_allow_html=True)
                with st.expander("Show Abstract"):
                    st.write(row["Abstract"])
                st.markdown("---")

            st.download_button(
                label="Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="studies_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No studies found for the given query and sources.")

with tab2:
    st.header("AI Overview")
    question = st.text_area("Ask a question related to your search:")

    if st.button("Generate AI Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Fetching studies for AI context..."):
                combined_docs = []
                if "PubMed" in source_filter:
                    combined_docs.extend(fetch_pubmed(query, max_results))
                if "NASA ADS" in source_filter:
                    nasa_token = st.secrets.get("NASA_ADS_API_TOKEN")
                    combined_docs.extend(fetch_nasa_ads(query, max_results, nasa_token))
            if combined_docs:
                with st.spinner("Generating AI answer..."):
                    answer = generate_ai_answer(question, combined_docs)
                    st.markdown("### AI-generated Answer")
                    st.write(answer)
            else:
                st.warning("No studies found to generate AI answer.")
