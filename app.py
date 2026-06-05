"""
Applicazione Streamlit completa per l'analisi di brevetti nel settore pressofusione.
Include tutte le funzionalità: dashboard, ricerca, trend, competitor, clustering NLP, risk analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from io import BytesIO
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import sqlite3
import time
import re
from contextlib import contextmanager
from abc import ABC, abstractmethod
from urllib.parse import quote

# --- Dipendenze di terze parti ---
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import umap.umap_ as umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import tempfile

# Download risorse NLTK (solo la prima volta)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# -------------------------------------------------------------------
# 1. CONFIGURAZIONE
# -------------------------------------------------------------------
class Settings:
    espacenet_consumer_key = None
    espacenet_consumer_secret = None
    uspto_api_key = None
    lens_bearer_token = None
    cache_ttl_hours = 24
    cache_db_path = Path("data/patent_cache.db")
    rate_limit_espacenet = 0.5
    rate_limit_uspto = 2.0
    rate_limit_lens = 1.0
    rate_limit_wipo = 0.2
    rate_limit_google = 0.1
    data_dir = Path("data")
    demo_data_dir = Path("demo_data")
    logs_dir = Path("logs")
    queries = {
        "Zama": {
            "keywords": '"zinc alloy die casting" OR "zamak" OR "zinc pressure casting"',
            "cpc": "B22D17/* AND C22C18/*"
        },
        "Alluminio": {
            "keywords": '"aluminium die casting" OR "aluminum pressure die casting"',
            "cpc": "B22D17/* AND C22C21/*"
        },
        "Magnesio": {
            "keywords": '"magnesium die casting" OR "magnesium alloy pressure casting"',
            "cpc": "B22D17/* AND C22C23/*"
        }
    }
    default_year_start = 2010
    default_year_end = 2025

settings = Settings()
settings.data_dir.mkdir(exist_ok=True)
settings.demo_data_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 2. MODELLI DATI (Pydantic)
# -------------------------------------------------------------------
class Applicant(BaseModel):
    name: str
    country: Optional[str] = None

class Patent(BaseModel):
    id: str
    title: str
    abstract: str
    filing_date: Optional[date] = None
    inventors: List[str] = Field(default_factory=list)
    applicants: List[Applicant] = Field(default_factory=list)
    cpc_classes: List[str] = Field(default_factory=list)
    status: str = "unknown"
    country_code: str = "EP"
    material_category: str
    data_source: str = "demo"

    @field_validator("material_category")
    @classmethod
    def validate_material(cls, v):
        if v not in {"Zama","Alluminio","Magnesio"}:
            raise ValueError(f"Materiale {v} non valido")
        return v

class FTOScore(BaseModel):
    patent_id: str
    score: float
    risk_level: str

# -------------------------------------------------------------------
# 3. CACHE SQLITE
# -------------------------------------------------------------------
class PatentCache:
    def __init__(self):
        self.db_path = settings.cache_db_path
        self.ttl_seconds = settings.cache_ttl_hours * 3600
        self._init_db()
    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS api_cache (key TEXT PRIMARY KEY, data TEXT, expires_at TIMESTAMP)")
            conn.commit()
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    def get(self, key):
        with self._get_connection() as conn:
            row = conn.execute("SELECT data, expires_at FROM api_cache WHERE key = ?", (key,)).fetchone()
            if row and datetime.fromisoformat(row["expires_at"]) > datetime.now():
                return json.loads(row["data"])
        return None
    def set(self, key, data):
        with self._get_connection() as conn:
            expires = (datetime.now() + timedelta(seconds=self.ttl_seconds)).isoformat()
            conn.execute("INSERT OR REPLACE INTO api_cache VALUES (?,?,?)", (key, json.dumps(data, default=str), expires))
            conn.commit()

cache = PatentCache()

# -------------------------------------------------------------------
# 4. FETCHER API (semplificati, con fallback a demo)
# -------------------------------------------------------------------
class BaseFetcher(ABC):
    @property
    @abstractmethod
    def source_name(self): pass
    @abstractmethod
    def fetch(self, material, year_start, year_end): pass

class EspacenetFetcher(BaseFetcher):
    source_name = "espacenet"
    def fetch(self, material, year_start, year_end):
        # Simulazione dati demo per brevità (in produzione usare OAuth)
        return []

class USPTOFetcher(BaseFetcher):
    source_name = "uspto"
    def fetch(self, material, year_start, year_end):
        return []

class LensFetcher(BaseFetcher):
    source_name = "lens"
    def fetch(self, material, year_start, year_end):
        return []

class WIPOFetcher(BaseFetcher):
    source_name = "wipo"
    def fetch(self, material, year_start, year_end):
        return []

class GooglePatentsScraper(BaseFetcher):
    source_name = "google"
    def fetch(self, material, year_start, year_end):
        return []

class DemoFetcher(BaseFetcher):
    source_name = "demo"
    def fetch(self, material, year_start, year_end):
        demo_data = [
            {"id": "EP1234567A1", "title": "Processo di pressofusione per leghe di Zama", "abstract": "Metodo per pressofusione di zinco", "filing_date": "2021-03-15", "inventors": ["Rossi M."], "applicants": [{"name": "Fonderie S.p.A.", "country": "IT"}], "cpc_classes": ["B22D17/22"], "status": "granted", "country_code": "EP", "material_category": "Zama", "filing_year": 2021},
            {"id": "US2022123456A1", "title": "Aluminium die casting method", "abstract": "High-pressure die casting of aluminium", "filing_date": "2020-06-20", "inventors": ["Johnson P."], "applicants": [{"name": "AutoTech GmbH", "country": "DE"}], "cpc_classes": ["B22D17/20"], "status": "granted", "country_code": "US", "material_category": "Alluminio", "filing_year": 2020},
            {"id": "WO2023123456A1", "title": "Magnesium alloy die casting", "abstract": "Process for magnesium alloys", "filing_date": "2022-09-01", "inventors": ["Chen W."], "applicants": [{"name": "Magnesium Research Inst.", "country": "CN"}], "cpc_classes": ["B22D17/00"], "status": "pending", "country_code": "WO", "material_category": "Magnesio", "filing_year": 2022},
        ]
        patents = []
        for p in demo_data:
            if p["material_category"] == material and year_start <= p["filing_year"] <= year_end:
                patents.append(Patent(
                    id=p["id"], title=p["title"], abstract=p["abstract"],
                    filing_date=date.fromisoformat(p["filing_date"]) if p["filing_date"] else None,
                    inventors=p["inventors"], applicants=[Applicant(**a) for a in p["applicants"]],
                    cpc_classes=p["cpc_classes"], status=p["status"], country_code=p["country_code"],
                    material_category=material, data_source="demo"
                ))
        return patents

class CompositeFetcher:
    def __init__(self):
        self.fetchers = [DemoFetcher()]  # solo demo per semplicità, ma puoi aggiungere gli altri
    def fetch_patents(self, material, year_start, year_end, force_refresh=False):
        key = f"{material}_{year_start}_{year_end}"
        if not force_refresh:
            cached = cache.get(key)
            if cached:
                return [Patent(**p) for p in cached]
        all_patents = []
        for f in self.fetchers:
            patents = f.fetch(material, year_start, year_end)
            if patents:
                all_patents.extend(patents)
                break
        cache.set(key, [p.model_dump(mode="json") for p in all_patents])
        return all_patents

fetcher = CompositeFetcher()

# -------------------------------------------------------------------
# 5. FUNZIONI DI ANALISI (trends, clustering, risk)
# -------------------------------------------------------------------
def compute_yearly_counts(patents):
    data = [{"year": p.filing_date.year, "material": p.material_category} for p in patents if p.filing_date]
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["year","material","count"])
    return df.groupby(["year","material"]).size().reset_index(name="count")

def get_top_applicants(patents, n=10):
    apps = []
    for p in patents:
        for a in p.applicants:
            apps.append({"name": a.name, "material": p.material_category})
    df = pd.DataFrame(apps)
    if df.empty:
        return pd.DataFrame()
    top = df.groupby("name").size().reset_index(name="total").sort_values("total", ascending=False).head(n)
    mat_counts = df.groupby(["name","material"]).size().unstack(fill_value=0)
    return top.merge(mat_counts, left_on="name", right_index=True, how="left")

def cluster_patents(patents):
    texts = [f"{p.title}. {p.abstract}" for p in patents if p.abstract]
    if len(texts) < 3:
        return {"labels": [], "embedding": None}
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=min(5, len(texts)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf)
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(tfidf.toarray())
    return {"labels": labels, "embedding": embedding, "texts": texts, "ids": [p.id for p in patents]}

def compute_fto_scores(patents):
    scores = []
    for p in patents[:100]:
        age_factor = 0.5
        if p.filing_date:
            age_years = (date.today() - p.filing_date).days / 365.25
            age_factor = max(0.2, 1.0 - age_years/20)
        score = age_factor * 100
        risk = "alto" if score > 70 else "medio" if score > 40 else "basso"
        scores.append(FTOScore(patent_id=p.id, score=round(score,2), risk_level=risk))
    return scores

def cpc_heatmap(patents):
    data = []
    for p in patents:
        for cpc in p.cpc_classes:
            sub = cpc[:4]
            data.append({"subclass": sub, "material": p.material_category})
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["subclass","material"]).size().unstack(fill_value=0)

# -------------------------------------------------------------------
# 6. INTERFACCIA STREAMLIT (multi-pagina tramite menu laterale)
# -------------------------------------------------------------------
st.set_page_config(page_title="Die Casting Patent Analytics", page_icon="🏭", layout="wide")

# Inizializza stato
if "patents" not in st.session_state:
    st.session_state.patents = []
if "filters" not in st.session_state:
    st.session_state.filters = {
        "material": ["Zama","Alluminio","Magnesio"],
        "year_range": (2010,2025),
        "status": ["granted","pending","expired","unknown"]
    }

# Sidebar per filtri e refresh
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/patent.png", width=60)
    st.title("🔍 Filtri")
    mat_filter = st.multiselect("Materiale", ["Zama","Alluminio","Magnesio"], default=st.session_state.filters["material"])
    year_range = st.slider("Anni", 2000,2030, st.session_state.filters["year_range"], step=1)
    status_filter = st.multiselect("Stato", ["granted","pending","expired","unknown"], default=st.session_state.filters["status"])
    refresh = st.button("🔄 Aggiorna dati")
    st.session_state.filters["material"] = mat_filter
    st.session_state.filters["year_range"] = year_range
    st.session_state.filters["status"] = status_filter

# Caricamento dati
if refresh or not st.session_state.patents:
    with st.spinner("Caricamento brevetti..."):
        all_p = []
        for mat in st.session_state.filters["material"]:
            all_p.extend(fetcher.fetch_patents(mat, year_range[0], year_range[1]))
        st.session_state.patents = all_p

# Applica filtro status
patents = [p for p in st.session_state.patents if p.status in status_filter]
st.sidebar.write(f"**Brevetti visualizzati:** {len(patents)}")

# Menu di navigazione
menu = st.sidebar.radio("Vai a:", ["Dashboard", "Ricerca", "Trend", "Competitor", "Clustering NLP", "Risk & White Space"])

# ------------------------------ PAGINA DASHBOARD ------------------------------
if menu == "Dashboard":
    st.title("📊 Dashboard Panoramica")
    if not patents:
        st.warning("Nessun brevetto.")
    else:
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Totale", len(patents))
        col2.metric("Zama", sum(1 for p in patents if p.material_category=="Zama"))
        col3.metric("Alluminio", sum(1 for p in patents if p.material_category=="Alluminio"))
        col4.metric("Magnesio", sum(1 for p in patents if p.material_category=="Magnesio"))

        yearly = compute_yearly_counts(patents)
        if not yearly.empty:
            fig = px.line(yearly, x="year", y="count", color="material", markers=True, title="Andamento depositi")
            st.plotly_chart(fig, use_container_width=True)

        top_apps = get_top_applicants(patents, n=8)
        if not top_apps.empty:
            fig2 = px.bar(top_apps, x="total", y="name", orientation="h", title="Top depositanti")
            st.plotly_chart(fig2, use_container_width=True)

# ------------------------------ PAGINA RICERCA ------------------------------
elif menu == "Ricerca":
    st.title("🔎 Ricerca Brevetti")
    if not patents:
        st.warning("Nessun dato.")
    else:
        df = pd.DataFrame([{
            "ID": p.id, "Titolo": p.title, "Abstract": p.abstract[:150],
            "Inventori": ", ".join(p.inventors[:2]), "Depositante": p.applicants[0].name if p.applicants else "",
            "Anno": p.filing_date.year if p.filing_date else "", "Materiale": p.material_category
        } for p in patents])
        search = st.text_input("Cerca per parola chiave")
        if search:
            df = df[df["Titolo"].str.contains(search, case=False) | df["Abstract"].str.contains(search, case=False)]
        st.dataframe(df, use_container_width=True)

# ------------------------------ PAGINA TREND ------------------------------
elif menu == "Trend":
    st.title("📈 Analisi Trend")
    if not patents:
        st.warning("Nessun dato.")
    else:
        yearly = compute_yearly_counts(patents)
        if not yearly.empty:
            fig = px.line(yearly, x="year", y="count", color="material", title="Numero depositi per anno")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------ PAGINA COMPETITOR ------------------------------
elif menu == "Competitor":
    st.title("🏢 Competitor Intelligence")
    if patents:
        top = get_top_applicants(patents, n=10)
        if not top.empty:
            fig = px.bar(top, x="total", y="name", orientation="h", title="Top 10 depositanti")
            st.plotly_chart(fig, use_container_width=True)

        # Grafo collaborazioni (semplice)
        G = nx.Graph()
        for p in patents[:50]:
            invs = p.inventors
            for i in range(len(invs)):
                for j in range(i+1, len(invs)):
                    G.add_edge(invs[i], invs[j])
        if G.number_of_nodes() > 0:
            net = Network(height="400px", width="100%", bgcolor="#222", font_color="white")
            net.from_nx(G)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                net.save_graph(f.name)
                with open(f.name) as html_file:
                    st.components.v1.html(html_file.read(), height=450)

# ------------------------------ PAGINA CLUSTERING ------------------------------
elif menu == "Clustering NLP":
    st.title("🧠 Technology Clustering")
    if len(patents) < 5:
        st.warning("Servono almeno 5 brevetti.")
    else:
        with st.spinner("Clustering in corso..."):
            res = cluster_patents(patents)
        if res["embedding"] is not None:
            df = pd.DataFrame({"x": res["embedding"][:,0], "y": res["embedding"][:,1], "cluster": res["labels"].astype(str), "id": res["ids"]})
            fig = px.scatter(df, x="x", y="y", color="cluster", hover_data=["id"], title="Cluster (UMAP)")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------ PAGINA RISK ------------------------------
else:  # Risk & White Space
    st.title("⚠️ IP Risk & White Space")
    if not patents:
        st.warning("Nessun brevetto.")
    else:
        scores = compute_fto_scores(patents)
        df_scores = pd.DataFrame([s.model_dump() for s in scores])
        st.subheader("Freedom-to-Operate Score")
        st.dataframe(df_scores.sort_values("score", ascending=False).head(10))

        heat = cpc_heatmap(patents)
        if not heat.empty:
            st.subheader("Copertura tecnologica per sottoclasse CPC")
            fig = px.imshow(heat, text_auto=True, aspect="auto", labels=dict(x="Materiale", y="CPC"))
            st.plotly_chart(fig, use_container_width=True)