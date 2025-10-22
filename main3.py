# For internal R&D decision support only — not clinical advice.

from sqlite3 import connect as sqlite_connect
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import json
import os
import re
import requests
from pydantic import BaseModel, Field

# --- LlamaIndex & Agents ---
from llama_index.core.tools import FunctionTool
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.llms.callbacks import llm_chat_callback

# --- Friendli AI ---
from friendli import SyncFriendli

# --- Weaviate ---
import weaviate
from weaviate.auth import Auth
from weaviate.agents.query import QueryAgent
from weaviate.classes.config import Configure, Property, DataType

# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------

# Weaviate connection (required)
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
if not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_API_KEY environment variable is required but not set")
if not WEAVIATE_URL:
    raise ValueError("WEAVIATE_URL environment variable is required but not set")

# FriendliAI configuration
FRIENDLI_MODEL = os.environ.get("FRIENDLI_MODEL", "meta-llama-3.1-8B-instruct")
# FRIENDLI_TEAM is hardcoded below in the FriendliLLM initialization

# Custom Friendli LLM wrapper for LlamaIndex
class FriendliLLM(LLM):
    def __init__(self, token: str, model: str, team: str, **kwargs):
        super().__init__(**kwargs)
        self._client = SyncFriendli(token=token)
        self._model = model
        self._team = team

    @property
    def metadata(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            is_function_calling_model=True,  # Assume it supports function calling
            context_window=32768,  # Default context window
            num_output=4096,  # Default max output tokens
        )

    @llm_chat_callback()
    def chat(self, messages, **kwargs):
        # Convert LlamaIndex messages to Friendli format
        friendli_messages = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                friendli_messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                friendli_messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.SYSTEM:
                friendli_messages.append({"role": "system", "content": msg.content})

        # Call Friendli API
        response = self._client.chat.completions.create(
            model=self._model,
            messages=friendli_messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            x_friendli_team=self._team,
        )

        # Convert response back to LlamaIndex format
        content = response.choices[0].message.content
        return ChatMessage(role=MessageRole.ASSISTANT, content=content)

    def complete(self, prompt, **kwargs):
        # For simple completion, wrap in a chat message
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = self.chat(messages, **kwargs)
        return response.content

    # Async methods (required by abstract base class)
    async def achat(self, messages, **kwargs):
        # For now, just call the sync version
        # In a real implementation, you'd want to use async Friendli client
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt, **kwargs):
        # For simple completion, wrap in a chat message
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = await self.achat(messages, **kwargs)
        return response.content

    # Streaming methods (required by abstract base class)
    def stream_chat(self, messages, **kwargs):
        # For now, return a single response as a generator
        # In a real implementation, you'd want to use streaming from Friendli
        response = self.chat(messages, **kwargs)
        yield response

    def stream_complete(self, prompt, **kwargs):
        # For simple completion, wrap in a chat message
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        for response in self.stream_chat(messages, **kwargs):
            yield response.content

    async def astream_chat(self, messages, **kwargs):
        # For now, return a single response as a generator
        # In a real implementation, you'd want to use async streaming from Friendli
        response = await self.achat(messages, **kwargs)
        yield response

    async def astream_complete(self, prompt, **kwargs):
        # For simple completion, wrap in a chat message
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        async for response in self.astream_chat(messages, **kwargs):
            yield response.content

# Initialize the Friendli LLM
llm = FriendliLLM(token="flp_DMcypemHFEKjm0qZqmZrZgkB1rqoAEAGB9BYiDjmaWjw18", model="dep30dsnycth0w2", team="4sGGjIpb9Aln")

# --------------------------------------------------------------------------------------
# Pydantic data classes for structured output
# --------------------------------------------------------------------------------------

class EvidenceSource(BaseModel):
    source: str = Field(description="Source of the evidence")
    title: str = Field(description="Title of the evidence")
    url: Optional[str] = Field(default="", description="URL of the evidence")

class Recommendation(BaseModel):
    compounds: List[str] = Field(description="Compound names under consideration")
    risk_level: str = Field(description="Overall interaction risk level", pattern="^(Low|Moderate|High|Unknown)$")
    rationale: str = Field(description="Rationale for the risk assessment")
    evidence: List[EvidenceSource] = Field(description="Sources of evidence used in the analysis")
    suggested_next_experiment: str = Field(description="Suggested next experiment")

# --------------------------------------------------------------------------------------
# Weaviate schema setup
# --------------------------------------------------------------------------------------

def connect_weaviate() -> weaviate.WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

def fresh_setup_weaviate(client: weaviate.WeaviateClient) -> QueryAgent:
    """
    DANGEROUS IN DEV — Drops and recreates the three domain collections.
    Returns a Weaviate QueryAgent over Compounds, RnDRecords, InteractionEvidence.
    """
    for name in ["Compounds", "RnDRecords", "InteractionEvidence"]:
        if client.collections.exists(name):
            client.collections.delete(name)

    # Compounds
    client.collections.create(
        "Compounds",
        description="Reference info for small molecules/biologics used in internal R&D.",
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        properties=[
            Property(name="name",         data_type=DataType.TEXT,       description="Canonical compound name"),
            Property(name="smiles",       data_type=DataType.TEXT,       description="SMILES string"),
            Property(name="synonyms",     data_type=DataType.TEXT_ARRAY, description="Synonyms / aliases"),
            Property(name="moa",          data_type=DataType.TEXT,       description="Mechanism of action"),
            Property(name="risk_flags",   data_type=DataType.TEXT_ARRAY, description="Known flags (e.g., QT prolongation, CYP3A4 inhibitor)"),
            Property(name="notes",        data_type=DataType.TEXT,       description="Free-text notes / datasheet"),
        ],
    )

    # Internal R&D records
    client.collections.create(
        "RnDRecords",
        description="Internal R&D notes, assay readouts, and protocol summaries.",
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        properties=[
            Property(name="doc_id",       data_type=DataType.TEXT,       description="Internal ID or URL"),
            Property(name="title",        data_type=DataType.TEXT,       description="Record title"),
            Property(name="body",         data_type=DataType.TEXT,       description="Full text of the record"),
            Property(name="compounds",    data_type=DataType.TEXT_ARRAY, description="Related compounds by name"),
            Property(name="study_phase",  data_type=DataType.TEXT,       description="Preclinical / Phase I–IV / NA"),
            Property(name="tags",         data_type=DataType.TEXT_ARRAY, description="Keywords or folders"),
            Property(name="created_at",   data_type=DataType.DATE,       description="Creation date"),
            Property(name="source",       data_type=DataType.TEXT,       description="e.g., ELN, LIMS, Confluence"),
        ],
    )

    # External evidence
    client.collections.create(
        "InteractionEvidence",
        description="External evidence about interactions/toxicity (PubMed, ClinicalTrials.gov).",
        vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        properties=[
            Property(name="source",         data_type=DataType.TEXT,       description="PubMed or ClinicalTrials.gov"),
            Property(name="pmid",           data_type=DataType.TEXT,       description="PubMed ID if available"),
            Property(name="nct_id",         data_type=DataType.TEXT,       description="ClinicalTrials.gov NCT ID if available"),
            Property(name="title",          data_type=DataType.TEXT,       description="Article or trial title"),
            Property(name="abstract",       data_type=DataType.TEXT,       description="Abstract / summary text"),
            Property(name="outcome",        data_type=DataType.TEXT,       description="Key outcomes related to safety/interaction"),
            Property(name="risk_signals",   data_type=DataType.TEXT_ARRAY, description="Adverse events / black-box warnings / contraindications"),
            Property(name="compounds",      data_type=DataType.TEXT_ARRAY, description="Mentioned compounds"),
            Property(name="published_date", data_type=DataType.DATE,       description="Publication date"),
            Property(name="url",            data_type=DataType.TEXT,       description="Source URL"),
        ],
    )

    # QueryAgent
    agent = QueryAgent(client=client, collections=["Compounds", "RnDRecords", "InteractionEvidence"])
    return agent

# --------------------------------------------------------------------------------------
# Ingestion helpers
# --------------------------------------------------------------------------------------

def _title_from(url: str, fallback: str = "Webpage") -> str:
    if not url:
        return fallback
    last = url.rstrip("/").rsplit("/", 1)[-1]
    return last or fallback

def write_webpages_to_weaviate(
    client: weaviate.WeaviateClient,
    urls: List[str],
    collection_name: Optional[str] = None,
):
    """Route PubMed/CT.gov to InteractionEvidence; other URLs to RnDRecords."""
    from llama_index.readers.web import SimpleWebPageReader
    docs = SimpleWebPageReader(html_to_text=True).load_data(urls)

    rnd = client.collections.get("RnDRecords")
    ev  = client.collections.get("InteractionEvidence")

    with rnd.batch.dynamic() as rnd_batch, ev.batch.dynamic() as ev_batch:
        for doc in docs:
            meta = doc.metadata or {}
            url = meta.get("url", "")
            title = meta.get("title") or _title_from(url)
            text = (doc.text or "")[:4000]

            if collection_name == "RnDRecords":
                rnd_batch.add_object(properties={
                    "doc_id": url or title,
                    "title": title,
                    "body": text,
                    "compounds": [],
                    "study_phase": "NA",
                    "tags": ["web_content", "external"],
                    "created_at": "2024-01-01T00:00:00Z",
                    "source": "Web",
                })
                continue

            if collection_name == "InteractionEvidence":
                ev_batch.add_object(properties={
                    "source": "Web",
                    "pmid": "",
                    "nct_id": "",
                    "title": title,
                    "abstract": text,
                    "outcome": "",
                    "risk_signals": [],
                    "compounds": [],
                    "published_date": "2024-01-01T00:00:00Z",
                    "url": url,
                })
                continue

            # Auto-route when unspecified
            if "pubmed.ncbi.nlm.nih.gov" in url:
                pmid = (re.search(r"/(\d+)/?$", url).group(1) if re.search(r"/(\d+)/?$", url) else "")
                ev_batch.add_object(properties={
                    "source": "PubMed",
                    "pmid": pmid,
                    "nct_id": "",
                    "title": title,
                    "abstract": text,
                    "outcome": "",
                    "risk_signals": [],
                    "compounds": [],
                    "published_date": "2024-01-01T00:00:00Z",
                    "url": url,
                })
            elif "clinicaltrials.gov" in url:
                nct = ((re.search(r"(NCT\d+)", url, re.IGNORECASE).group(1).upper())
                       if re.search(r"(NCT\d+)", url, re.IGNORECASE) else "")
                ev_batch.add_object(properties={
                    "source": "ClinicalTrials.gov",
                    "pmid": "",
                    "nct_id": nct,
                    "title": title,
                    "abstract": text,
                    "outcome": "",
                    "risk_signals": [],
                    "compounds": [],
                    "published_date": "2024-01-01T00:00:00Z",
                    "url": url,
                })
            else:
                rnd_batch.add_object(properties={
                    "doc_id": url or title,
                    "title": title,
                    "body": text,
                    "compounds": [],
                    "study_phase": "NA",
                    "tags": ["web_content", "external"],
                    "created_at": "2024-01-01T00:00:00Z",
                    "source": "Web",
                })

# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------

def make_weaviate_query_tool(client: weaviate.WeaviateClient) -> FunctionTool:
    qa = QueryAgent(client=client, collections=["Compounds", "RnDRecords", "InteractionEvidence"])
    def run(query: str) -> str:
        return qa.run(query).final_answer
    return FunctionTool.from_defaults(
        fn=run,
        name="weaviate_query",
        description="Query Compounds, RnDRecords, and InteractionEvidence in Weaviate."
    )

# --- PubMed (NCBI E-utilities) ---

def pubmed_search_and_fetch(query: str, retmax: int = 5) -> List[Dict[str, str]]:
    """
    Returns a list of {source, title, url, pmid} using ESearch + ESummary.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    es = requests.get(f"{base}/esearch.fcgi", params={"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}, timeout=30)
    es.raise_for_status()
    ids = es.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    sm = requests.get(f"{base}/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=30)
    sm.raise_for_status()
    data = sm.json().get("result", {})
    out = []
    for pmid in ids:
        rec = data.get(pmid, {})
        title = rec.get("title") or f"PMID {pmid}"
        out.append({
            "source": "PubMed",
            "title": title,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "pmid": pmid,
        })
    return out

def make_pubmed_tool() -> FunctionTool:
    def run(query: str) -> List[Dict[str, str]]:
        return pubmed_search_and_fetch(query)
    return FunctionTool.from_defaults(
        fn=run,
        name="pubmed_eutils",
        description="Search PubMed for interactions/toxicity evidence; returns list of citations."
    )

# --- ClinicalTrials.gov v2 ---

def ctgov_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Returns a list of {source, title, url, nct_id}.
    """
    # v2 API: simple query by term; keep robust with fallback
    url = "https://clinicaltrials.gov/api/v2/studies"
    resp = requests.get(url, params={"format":"json", "query.term": query, "pageSize": limit}, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    out: List[Dict[str, str]] = []
    studies = js.get("studies", []) or js.get("studies", [])
    for s in studies[:limit]:
        prot = s.get("protocolSection", {})
        ident = prot.get("identificationModule", {})
        nct_id = ident.get("nctId")
        title = ident.get("briefTitle") or f"Study {nct_id or ''}".strip()
        if nct_id:
            out.append({
                "source": "ClinicalTrials.gov",
                "title": title,
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "nct_id": nct_id,
            })
    return out

def make_ctgov_tool() -> FunctionTool:
    def run(query: str) -> List[Dict[str, str]]:
        return ctgov_search(query)
    return FunctionTool.from_defaults(
        fn=run,
        name="clinicaltrials_v2",
        description="Search ClinicalTrials.gov for trials mentioning the compounds; returns list of citations."
    )

# --- Toxicology rules (very simple, keyword-based over retrieved context) ---

TOX_KEYWORDS = {
    "qt": ["qt prolongation", "torsades", "long qt"],
    "cns": ["sedation", "cns depression", "respiratory depression"],
    "cyp_inhib": ["cyp3a4 inhibitor", "cyp2d6 inhibitor", "strong inhibitor"],
    "cyp_sub":   ["cyp3a4 substrate", "cyp2d6 substrate"]
}

def heuristic_tox_score(compounds: List[str], context_blobs: List[str]) -> Dict[str, Any]:
    """
    Inspect combined text for crude risk signals. Returns {risk_level, reasons}.
    This is a placeholder; replace with your real rules engine.
    """
    text = " ".join([c.lower() for c in context_blobs if c])
    reasons: List[str] = []
    found = {k: any(any(term in text for term in vals) for k2, vals in {k: v}.items()) for k, v in TOX_KEYWORDS.items()}

    # Sample heuristics
    if found["qt"]:
        reasons.append("Signals of QT prolongation found.")
    if found["cns"]:
        reasons.append("Signals of additive CNS depression found.")
    if found["cyp_inhib"] and found["cyp_sub"]:
        reasons.append("CYP inhibitor + substrate overlap suggests exposure increase.")

    if not reasons:
        risk = "Unknown" if not text.strip() else "Low"
    elif any("QT" in r or "qt" in r for r in reasons):
        risk = "High"
    else:
        risk = "Moderate"

    return {"risk_level": risk, "reasons": reasons or ["Insufficient explicit signals; defaulting to conservative assessment."]}

def make_tox_tool() -> FunctionTool:
    def run(compounds_csv_or_query: str, context: Optional[str] = None) -> Dict[str, Any]:
        # allow passing a comma-separated list of compounds and a context blob
        comps = [c.strip() for c in re.split(r"[;,/]", compounds_csv_or_query) if c.strip()]
        blobs = [context] if context else []
        return heuristic_tox_score(comps, blobs)
    return FunctionTool.from_defaults(
        fn=run,
        name="tox_rules",
        description="Apply simple tox heuristics over provided context to estimate interaction risk."
    )

# --- Experiment logger (SQLite) ---

def log_recommendation(rec: Dict[str, Any], db_path: str = "advisor_logs.db") -> str:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite_connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            compounds TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            rationale TEXT NOT NULL,
            evidence TEXT NOT NULL,
            next_experiment TEXT NOT NULL,
            raw_json TEXT NOT NULL
        )
    """)
    con.commit()
    cur.execute("""
        INSERT INTO recommendations (created_at, compounds, risk_level, rationale, evidence, next_experiment, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        json.dumps(rec.get("compounds", [])),
        rec.get("risk_level", "Unknown"),
        rec.get("rationale", ""),
        json.dumps(rec.get("evidence", [])),
        rec.get("suggested_next_experiment", ""),
        json.dumps(rec),
    ))
    con.commit()
    rowid = cur.lastrowid
    con.close()
    return f"logged_recommendation_{rowid}"

def make_logger_tool() -> FunctionTool:
    def run(rec_json: Dict[str, Any]) -> str:
        return log_recommendation(rec_json)
    return FunctionTool.from_defaults(
        fn=run,
        name="experiment_logger",
        description="Persist the final recommendation to SQLite."
    )

# --------------------------------------------------------------------------------------
# Orchestrator (simple, imperative glue)
# --------------------------------------------------------------------------------------

def run_advisor(
    client: weaviate.WeaviateClient,
    compounds: List[str],
    question: str,
    topk_external: int = 5
) -> Dict[str, Any]:
    """
    1) Query Weaviate (internal + cached external)
    2) Query PubMed & CT.gov
    3) Run tox heuristics
    4) Synthesize a structured Recommendation with the LLM
    5) Log to SQLite
    """
    # 1) Internal (and whatever has been ingested) via QueryAgent
    qa = QueryAgent(client=client, collections=["Compounds", "RnDRecords", "InteractionEvidence"])
    internal_prompt = (
        f"Summarize known interaction/safety signals for: {', '.join(compounds)}. "
        "List any URLs, PMIDs, or NCT IDs if present."
    )
    internal_answer = qa.run(internal_prompt).final_answer

    # 2) External evidence
    query_terms = " AND ".join([f'"{c}"' for c in compounds]) + " AND (interaction OR toxicity OR adverse)"
    pubmed_hits = pubmed_search_and_fetch(query_terms, retmax=topk_external)
    ctgov_hits  = ctgov_search(" ".join(compounds) + " interaction OR adverse", limit=topk_external)

    # 3) Tox heuristics over combined text context
    ext_text = "\n".join([h.get("title", "") for h in pubmed_hits + ctgov_hits])
    tox = heuristic_tox_score(compounds, [internal_answer, ext_text])

    # 4) Synthesize final JSON with FunctionCallingProgram (schema-enforced)
    synth = FunctionCallingProgram.from_defaults(
        output_cls=Recommendation,
        llm=llm,
        prompt_template_str="For internal R&D decision support only — not clinical advice.\nGiven the following context, return a Recommendation JSON.\n\nCompounds: {compounds}\n\nInternal retrieval:\n{internal_answer}\n\nExternal evidence (titles & links):\n{evidence_dicts}\n\nToxicology rules result:\n{tox}\n\nUse conservative language. Always include a suggested_next_experiment that is safe and incremental."
    )
    evidence_for_llm = [
        EvidenceSource(source=h["source"], title=h["title"], url=h.get("url", ""))
        for h in (pubmed_hits + ctgov_hits)
    ]
    
    # Convert evidence to dict for JSON serialization in prompt
    evidence_dicts = [{"source": e.source, "title": e.title, "url": e.url} for e in evidence_for_llm]
    
    final_json = synth(
        compounds=', '.join(compounds),
        internal_answer=internal_answer,
        evidence_dicts=json.dumps(evidence_dicts, indent=2),
        tox=json.dumps(tox, indent=2)
    )

    # 5) Log & return
    _log_id = log_recommendation(final_json)
    final_json["_log_id"] = _log_id
    return final_json

# --------------------------------------------------------------------------------------
# Demo / main
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Connecting to Weaviate…")
    client = connect_weaviate()
    try:
        # WARNING: destructive in dev
        print("Resetting schema (Compounds, RnDRecords, InteractionEvidence)…")
        _ = fresh_setup_weaviate(client)

        # Optional: ingest a couple of pages (swap with your real sources)
        demo_urls = [
            # Replace with relevant pages; left empty by default to avoid scraping on run
            # "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            # "https://clinicaltrials.gov/study/NCT04280705",
            "https://clinicaltrials.gov/study/NCT04280705"
        ]
        if demo_urls:
            print("Ingesting demo URLs…")
            write_webpages_to_weaviate(client, demo_urls)

        # Run an advisor call (replace with your compounds)
        compounds = ["Compound A", "Compound B"]
        print(f"Running advisor for: {compounds} …")
        rec = run_advisor(client, compounds, "Assess interaction risks.")
        print("\n=== Recommendation ===")
        print(json.dumps(rec, indent=2))

        print("\nSaved to SQLite (advisor_logs.db).")

    finally:
        client.close()
        print("Closed Weaviate client.")
