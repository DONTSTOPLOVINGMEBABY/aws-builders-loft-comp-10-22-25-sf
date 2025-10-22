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

import re 

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
    """Write webpage content to the appropriate pharmaceutical R&D collection"""
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    collection = client.collections.get(collection_name)
    
    with collection.batch.dynamic() as batch:
        for doc in documents:
            if collection_name == "RnDRecords":
                # For R&D records, map webpage content to R&D schema
                batch.add_object(properties={
                    "doc_id": doc.id_,
                    "title": doc.id_.split('/')[-1] or "Webpage Content",  # Extract title from URL
                    "body": doc.text,
                    "compounds": [],  # Will be populated by analysis
                    "study_phase": "NA",  # Default for web content
                    "tags": ["web_content", "external"],
                    "created_at": "2024-01-01T00:00:00Z",  # Default date
                    "source": "Web"
                })
            elif collection_name == "InteractionEvidence":
                # For interaction evidence, map to evidence schema
                batch.add_object(properties={
                    "source": "Web",
                    "pmid": "",
                    "nct_id": "",
                    "title": doc.id_.split('/')[-1] or "Webpage Content",
                    "abstract": doc.text[:1000] if len(doc.text) > 1000 else doc.text,  # Truncate if too long
                    "outcome": "",
                    "risk_signals": [],
                    "compounds": [],
                    "published_date": "2024-01-01T00:00:00Z",
                    "url": doc.id_
                })
            else:
                # For Compounds collection, this might not be appropriate
                # but we'll create a basic entry
                batch.add_object(properties={
                    "name": doc.id_.split('/')[-1] or "Unknown",
                    "smiles": "",
                    "synonyms": [],
                    "moa": "",
                    "risk_flags": [],
                    "notes": doc.text
                })





                

# # Define events for the pharmaceutical R&D workflow
# class EvaluateQuery(Event):
#     query: str

# class WriteRnDRecordsEvent(Event):
#     urls: list[str]

# class WriteInteractionEvidenceEvent(Event):
#     urls: list[str]

# class WriteCompoundsEvent(Event):
#     urls: list[str]

# class QueryAgentEvent(Event):
#     query: str

# class ActionCompleted(Event):
#     result: str

# # Define structured outputs for query classification
# class SaveToRnDRecords(BaseModel):
#     """URLs to parse and save into R&D records collection."""
#     rnd_urls: List[str] = Field(default_factory=list)

# class SaveToInteractionEvidence(BaseModel):
#     """URLs to parse and save into interaction evidence collection."""
#     evidence_urls: List[str] = Field(default_factory=list)

# class SaveToCompounds(BaseModel):
#     """URLs to parse and save into compounds collection."""
#     compound_urls: List[str] = Field(default_factory=list)

# class AskQuestion(BaseModel):
#     """Natural language questions for the QueryAgent."""
#     queries: List[str] = Field(default_factory=list)

# class Actions(BaseModel):
#     """Actions to take based on the user query."""
#     actions: List[
#         Union[SaveToRnDRecords, SaveToInteractionEvidence, SaveToCompounds, AskQuestion]
#     ] = Field(default_factory=list)

# # Pharmaceutical R&D Workflow
# class PharmaceuticalRDWorkflow(Workflow):
#     def __init__(self, client, weaviate_agent, *args, **kwargs):
#         self.client = client
#         self.weaviate_agent = weaviate_agent
#         self.llm = OpenAI(model="gpt-4o-mini")
#         self.system_prompt = """You are a pharmaceutical R&D assistant. You evaluate incoming queries and decide on the best course of action:
#         - Save URLs to R&D records collection (for internal research documents, protocols, assay results)
#         - Save URLs to interaction evidence collection (for PubMed articles, clinical trial data, safety reports)
#         - Save URLs to compounds collection (for compound databases, chemical information)
#         - Answer questions about compounds, R&D records, and interaction evidence using the QueryAgent
#         """
#         super().__init__(*args, **kwargs)

#     @step
#     async def start(self, ctx: Context, ev: StartEvent) -> EvaluateQuery:
#         return EvaluateQuery(query=ev.query)

#     @step
#     async def evaluate_query(
#         self, ctx: Context, ev: EvaluateQuery
#     ) -> QueryAgentEvent | WriteRnDRecordsEvent | WriteInteractionEvidenceEvent | WriteCompoundsEvent | None:
#         await ctx.store.set("results", [])
        
#         # Use structured LLM to classify the query
#         structured_llm = self.llm.as_structured_llm(Actions)
#         response = await structured_llm.achat([
#             ChatMessage(role="system", content=self.system_prompt),
#             ChatMessage(role="user", content=ev.query),
#         ])
        
#         actions = response.raw.actions
#         await ctx.store.set("num_events", len(actions))
#         await ctx.store.set("results", [])
        
#         print(f"Classified actions: {actions}")
        
#         for action in actions:
#             if isinstance(action, SaveToRnDRecords):
#                 ctx.send_event(WriteRnDRecordsEvent(urls=action.rnd_urls))
#             elif isinstance(action, SaveToInteractionEvidence):
#                 ctx.send_event(WriteInteractionEvidenceEvent(urls=action.evidence_urls))
#             elif isinstance(action, SaveToCompounds):
#                 ctx.send_event(WriteCompoundsEvent(urls=action.compound_urls))
#             elif isinstance(action, AskQuestion):
#                 for query in action.queries:
#                     ctx.send_event(QueryAgentEvent(query=query))

#     @step
#     async def write_rnd_records(
#         self, ctx: Context, ev: WriteRnDRecordsEvent
#     ) -> ActionCompleted:
#         print(f"Writing {ev.urls} to R&D Records")
#         write_webpages_to_weaviate(self.client, urls=ev.urls, collection_name="RnDRecords")
#         results = await ctx.store.get("results")
#         results.append(f"Wrote {ev.urls} to R&D Records")
#         return ActionCompleted(result=f"Writing {ev.urls} to R&D Records")

#     @step
#     async def write_interaction_evidence(
#         self, ctx: Context, ev: WriteInteractionEvidenceEvent
#     ) -> ActionCompleted:
#         print(f"Writing {ev.urls} to Interaction Evidence")
#         write_webpages_to_weaviate(self.client, urls=ev.urls, collection_name="InteractionEvidence")
#         results = await ctx.store.get("results")
#         results.append(f"Wrote {ev.urls} to Interaction Evidence")
#         return ActionCompleted(result=f"Writing {ev.urls} to Interaction Evidence")

#     @step
#     async def write_compounds(
#         self, ctx: Context, ev: WriteCompoundsEvent
#     ) -> ActionCompleted:
#         print(f"Writing {ev.urls} to Compounds")
#         write_webpages_to_weaviate(self.client, urls=ev.urls, collection_name="Compounds")
#         results = await ctx.store.get("results")
#         results.append(f"Wrote {ev.urls} to Compounds")
#         return ActionCompleted(result=f"Writing {ev.urls} to Compounds")

#     @step
#     async def query_agent(
#         self, ctx: Context, ev: QueryAgentEvent
#     ) -> ActionCompleted:
#         print(f"Querying agent with: {ev.query}")
#         response = self.weaviate_agent.run(ev.query)
#         results = await ctx.store.get("results")
#         results.append(f"QueryAgent responded: {response.final_answer}")
#         return ActionCompleted(result=f"QueryAgent response: {response.final_answer}")

#     @step
#     async def collect(
#         self, ctx: Context, ev: ActionCompleted
#     ) -> StopEvent | None:
#         num_events = await ctx.store.get("num_events")
#         evs = ctx.collect_events(ev, [ActionCompleted] * num_events)
#         if evs is None:
#             return None
#         return StopEvent(result=[ev.result for ev in evs])

# # Main execution function
# async def main():
#     """Main function to run the pharmaceutical R&D workflow"""
#     # Connect to Weaviate
#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url=WEAVIATE_URL,
#         auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
#     )
    
#     # Set up collections and get QueryAgent
#     weaviate_agent = fresh_setup_weaviate(client)
    
#     # Create and run workflow
#     workflow = PharmaceuticalRDWorkflow(client=client, weaviate_agent=weaviate_agent, timeout=None)
    
#     # Example usage
#     print("Pharmaceutical R&D Workflow Started!")
#     print("=" * 50)
    
#     # Example 1: Save some R&D documents
#     result1 = await workflow.run(
#         start_event=StartEvent(query="Save these R&D documents: https://example.com/protocol1, https://example.com/assay-results")
#     )
#     print("Result 1:", result1)
    
#     # Example 2: Query about compounds
#     result2 = await workflow.run(
#         start_event=StartEvent(query="What compounds do we have in our database?")
#     )
#     print("Result 2:", result2)
    
#     # Example 3: Save interaction evidence
#     result3 = await workflow.run(
#         start_event=StartEvent(query="Save this safety report: https://example.com/safety-report")
#     )
#     print("Result 3:", result3)
    
#     client.close()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

