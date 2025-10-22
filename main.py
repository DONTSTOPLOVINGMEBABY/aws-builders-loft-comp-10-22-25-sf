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

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union
import json

import weaviate
from weaviate.auth import Auth
from weaviate.agents.query import QueryAgent
from weaviate.classes.config import Configure, Property, DataType

import os

# Read environment variables that will be injected at runtime
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")

# Validate that required environment variables are set
if not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_API_KEY environment variable is required but not set")
if not WEAVIATE_URL:
    raise ValueError("WEAVIATE_URL environment variable is required but not set")

# imports this notebook already uses
from weaviate.agents.query import QueryAgent
from weaviate.collections.classes.config import Property, DataType, Configure

def fresh_setup_weaviate(client):
    # Clean up any old demo collections
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
        client=client, collections=["Compounds", "RnDRecords", "InteractionEvidence"]
    )
    return agent

def write_webpages_to_weaviate(client, urls: list[str], collection_name: str):
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    collection = client.collections.get(collection_name)
    with collection.batch.dynamic() as batch:
        for doc in documents:
            batch.add_object(properties={"url": doc.id_, "text": doc.text})

