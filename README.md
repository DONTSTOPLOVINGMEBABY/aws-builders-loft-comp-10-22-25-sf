# Drug Interaction Advisor

A multi-agent system for analyzing drug interactions using Weaviate vector database, OpenAI, and external APIs (PubMed, ClinicalTrials.gov).

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini

# Weaviate Configuration  
WEAVIATE_API_KEY=your_weaviate_api_key_here
WEAVIATE_URL=your_weaviate_cluster_url_here
```

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables (see above)
3. Run: `python main.py`

## Features

- Semantic search across internal R&D data
- Real-time PubMed and ClinicalTrials.gov queries
- Automated toxicity risk assessment
- Structured recommendation generation
