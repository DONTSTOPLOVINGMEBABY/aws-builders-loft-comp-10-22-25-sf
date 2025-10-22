# For internal R&D decision support only — not clinical advice.

from sqlite3 import connect
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

from openai import OpenAI as OpenAI2

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import json
import os
import re

import weaviate
from weaviate.auth import Auth
from weaviate.agents.query import QueryAgent
from weaviate.classes.config import Configure, Property, DataType
from friendli import SyncFriendli, models



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

# FriendliAI (OpenAI-compatible) — optional here, but useful once you wire agents


# --------------------------------------------------------------------------------------
# Weaviate schema setup
# --------------------------------------------------------------------------------------

def fresh_setup_weaviate(client: weaviate.WeaviateClient) -> QueryAgent:
    """
    DANGEROUS IN DEV — Drops and recreates the three domain collections.
    Returns a Weaviate QueryAgent over Compounds, RnDRecords, InteractionEvidence.
    """
    # Clean up any old demo collections (WARNING: destructive)
    for name in ["Compounds", "RnDRecords", "InteractionEvidence"]:
        if client.collections.exists(name):
            client.collections.delete(name)

    # === Compounds ===
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

    # === Internal R&D records ===
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

    # === External evidence (PubMed / ClinicalTrials.gov) ===
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

    # QueryAgent over all three collections
    agent = QueryAgent(
        client=client,
        collections=["Compounds", "RnDRecords", "InteractionEvidence"],
    )
    return agent


# --------------------------------------------------------------------------------------
# Ingestion helpers
# --------------------------------------------------------------------------------------

def write_webpages_to_weaviate(
    client: weaviate.WeaviateClient,
    urls: List[str],
    collection_name: Optional[str] = None,
):
    """
    Ingest webpages using SimpleWebPageReader and store them appropriately:
      - If collection_name is "InteractionEvidence": write as generic external evidence objects.
      - If collection_name is "RnDRecords": write as external web records.
      - If collection_name is None or unrecognized: auto-route
            PubMed   -> InteractionEvidence (extract pmid)
            CT.gov   -> InteractionEvidence (extract nct_id)
            other    -> RnDRecords
      - Will NOT write to Compounds (that collection should be populated from structured sources).

    NOTE: We rely on doc.metadata["url"] for the true URL; doc.id_ is a UUID.
    """

    docs = SimpleWebPageReader(html_to_text=True).load_data(urls)

    # Collections
    rnd = client.collections.get("RnDRecords")
    ev  = client.collections.get("InteractionEvidence")

    # Helpers
    def _title_from(url: str, fallback: str = "Webpage") -> str:
        if not url:
            return fallback
        last = url.rstrip("/").rsplit("/", 1)[-1]
        return last or fallback

    def _insert_rnd(url: str, title: str, text: str):
        rnd.data.insert({
            "doc_id": url or title,
            "title": title,
            "body": text,
            "compounds": [],
            "study_phase": "NA",
            "tags": ["web_content", "external"],
            "created_at": "2024-01-01T00:00:00Z",
            "source": "Web",
        })

    def _insert_pubmed(url: str, title: str, text: str):
        m = re.search(r"/(\d+)/?$", url)
        pmid = m.group(1) if m else ""
        ev.data.insert({
            "source": "PubMed",
            "pmid": pmid,
            "nct_id": "",
            "title": title,
            "abstract": text[:4000],
            "outcome": "",
            "risk_signals": [],
            "compounds": [],
            "published_date": "2024-01-01T00:00:00Z",
            "url": url,
        })

    def _insert_ctgov(url: str, title: str, text: str):
        m = re.search(r"(NCT\d+)", url, re.IGNORECASE)
        nct_id = m.group(1).upper() if m else ""
        ev.data.insert({
            "source": "ClinicalTrials.gov",
            "pmid": "",
            "nct_id": nct_id,
            "title": title,
            "abstract": text[:4000],
            "outcome": "",
            "risk_signals": [],
            "compounds": [],
            "published_date": "2024-01-01T00:00:00Z",
            "url": url,
        })

    # Batch insert dynamically for speed
    with rnd.batch.dynamic() as rnd_batch, ev.batch.dynamic() as ev_batch:
        for doc in docs:
            meta = doc.metadata or {}
            url = meta.get("url", "")
            title = meta.get("title") or _title_from(url)
            text = doc.text or ""

            # Explicit routing if collection_name is provided (and supported)
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
                    "abstract": text[:4000],
                    "outcome": "",
                    "risk_signals": [],
                    "compounds": [],
                    "published_date": "2024-01-01T00:00:00Z",
                    "url": url,
                })
                continue

            if collection_name == "Compounds":
                # Not supported: don't write arbitrary web pages to Compounds
                raise ValueError("Refusing to write generic web pages into 'Compounds'.")

            # Auto-routing when no explicit collection_name given
            if "pubmed.ncbi.nlm.nih.gov" in url:
                ev_batch.add_object(properties={
                    "source": "PubMed",
                    "pmid": (re.search(r"/(\d+)/?$", url).group(1) if re.search(r"/(\d+)/?$", url) else ""),
                    "nct_id": "",
                    "title": title,
                    "abstract": text[:4000],
                    "outcome": "",
                    "risk_signals": [],
                    "compounds": [],
                    "published_date": "2024-01-01T00:00:00Z",
                    "url": url,
                })
            elif "clinicaltrials.gov" in url:
                ev_batch.add_object(properties={
                    "source": "ClinicalTrials.gov",
                    "pmid": "",
                    "nct_id": ((re.search(r"(NCT\d+)", url, re.IGNORECASE).group(1).upper())
                               if re.search(r"(NCT\d+)", url, re.IGNORECASE) else ""),
                    "title": title,
                    "abstract": text[:4000],
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
# Tooling: expose QueryAgent as a LlamaIndex tool for your orchestrator
# --------------------------------------------------------------------------------------

def make_weaviate_query_tool(client: weaviate.WeaviateClient) -> FunctionTool:
    """
    Wrap a Weaviate QueryAgent as a LlamaIndex FunctionTool so agents/workflows can call it.
    """
    qa = QueryAgent(client=client, collections=["Compounds", "RnDRecords", "InteractionEvidence"])

    def run(query: str) -> str:
        """
        Natural-language query over the three collections.
        Returns the agent's final_answer string (which may include citations/URLs if present in objects).
        """
        result = qa.run(query)
        return result.final_answer

    return FunctionTool.from_defaults(
        fn=run,
        name="weaviate_query",
        description="Query Compounds, RnDRecords, and InteractionEvidence in Weaviate."
    )


# --------------------------------------------------------------------------------------
# (Optional) Quick connection helper (not required if you connect elsewhere)
# --------------------------------------------------------------------------------------

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
)

weaviate_agent = fresh_setup_weaviate(client)

llm = OpenAI2(
    base_url="https://api.friendli.ai/dedicated/v1",
    api_key="flp_DMcypemHFEKjm0qZqmZrZgkB1rqoAEAGB9BYiDjmaWjw18"  # your Friendli token
)

FRIENDLY_TOKEN = "flp_DMcypemHFEKjm0qZqmZrZgkB1rqoAEAGB9BYiDjmaWjw18"

try:
    with SyncFriendli(token=FRIENDLY_TOKEN) as friendli:
        resp = friendli.dedicated.chat.complete(
            model="dep30dsnycth0w2",  # e.g. "zbimjgovmlcb" — your endpoint ID
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=200,
            x_friendli_team="4sGGjIpb9Aln"
        )
        print("✅ Connected successfully!")
        print("Response:", resp.choices[0].message.content)
except models.SDKError as e:
    print(f"❌ Connection failed: {e.status_code} {e.message}")
except Exception as e:
    print("❌ Connection failed:", e)


client.close()
