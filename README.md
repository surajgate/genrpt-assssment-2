# Intelligent Query Router - Hybrid RAG with PDF + SQL

This project implements an Intelligent Query Router that routes questions between a structured SQL database and unstructured PDF documents stored in Milvus. It leverages hybrid search (dense + sparse retrieval), OpenAI embeddings, and GPT-based reasoning.

---

## 🚀 Features

- **SQL Querying**: Uses LLMs to generate SQL queries from natural language.
- **PDF Vector Search**: Extracts text from PDF using OCR, chunks it, embeds it, and stores it in Milvus.
- **Hybrid Routing**: Intelligent routing between SQL and vector DB based on confidence scoring.
- **Rephrasing**: Automatically rephrases queries if confidence is low.

---

## 📦 Prerequisites

### 1. Clone the Repository

```bash
git clone https://github.com/surajgate/genrpt-assssment-2.git
cd  genrpt-assssment-2
git checkout main
```

### 2. Create and Configure `.env` File

Copy the sample:

```bash
cp .env.sample .env
```

Update the .env file with the following:

```bash
# Milvus settings
MILVUS_CONNECTION_URI=<your-milvus-connection-url>
COLLECTION_NAME=<your-milvus-collection-name>

# Hybrid search parameters
NUM_CHUNKS_RETRIEVED=15
NUM_CHUNKS_TO_HYBRID_SEARCH=100
DENSE_VECTOR_WEIGHT=0.65
SPARSE_VECTOR_WEIGHT=0.35

# PostgreSQL settings
DB_URL=<your-db-url>

OPENAI_API_KEY=<api-key>
```

### 3. Install System Packages

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
```

## Set Up Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## SQL Database Setup

Make sure you have a PostgreSQL database up and running.

Update your `.env` file with a valid PostgreSQL connection string in the following format:

```env
DB_URL=postgresql://<username>:<password>@<host>:<port>/<database_name>
```

## PDF Ingestion into Milvus

Make sure Milvus is running on the URL specified in the .env file.

To extract, chunk, embed, and store PDF data:

```bash
cd genrpt-assessment

python pdf_processing.py
```

This script will:

Convert PDF pages into high-DPI images

Run Tesseract OCR for text extraction

Chunk and vectorize text

Store vectors into Milvus using hybrid search configuration

## Running the Agent

``` bash
cd genrpt-assessment
```

Edit main.py and provide a list of questions:
``` bash
questions = [
    "What was the IPL final score in 2023?",
    "How many wickets did Bumrah take in Qualifier 1?",
    ...
]
```
Then run:
``` bash
python main.py
```

The agent will:

Determine if a question is best answered via SQL or Vector DB

Retrieve data accordingly

Rephrase and retry low-confidence queries if needed
