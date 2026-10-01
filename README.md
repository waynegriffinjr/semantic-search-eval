# Semantic Search Evaluation

A Python-based evaluation framework for measuring the performance of a semantic search system using **ChromaDB**, with a focus on precision, recall, retrieval settings, and distance-based filtering.

This project evaluates a semantic search collection against a predefined set of queries and expected relevant documents. It compares multiple retrieval configurations to understand how search settings affect retrieval quality.

## Overview

Semantic search retrieves information based on the meaning of text rather than relying solely on exact keyword matches.

Building a search system, however, is only part of the problem. A reliable search system also needs to be evaluated to determine whether it is returning relevant information.

This project establishes a small evaluation framework that measures:

* **Precision** — How many retrieved documents are relevant?
* **Recall** — How many relevant documents were successfully retrieved?
* The effect of increasing `n_results`
* The effect of distance-based filtering
* Query-specific retrieval performance
* Tradeoffs between precision and recall

The evaluation uses a predefined set of relevant document IDs as the ground truth.

---

## Technologies

* Python
* ChromaDB
* Vector embeddings
* Semantic search
* Information retrieval
* Precision and recall
* Distance-based filtering
* Python sets and collection operations

---

## Project Structure

```text
Semantic-Search-Evaluation/
│
├── starter.py
└── README.md
```

> The Python file can be renamed from `starter.py` to a more descriptive name such as `evaluate_search.py` if desired. The implementation itself is the completed evaluation system.

---

## Dataset

The evaluation collection contains **12 documents** covering multiple technical topics:

* JWT authentication
* Pydantic and FastAPI
* Streamlit session state
* CORS
* Embeddings
* Cosine similarity
* Chunking
* ChromaDB
* CSS Flexbox
* JavaScript DOM
* Prompt injection
* Prompt-injection defenses

Each document contains:

* A unique document ID
* Document content
* Topic metadata

The evaluation set contains **6 test queries**. Each query includes a list of document IDs that are considered relevant.

This creates a simple ground-truth dataset against which the semantic search results can be measured.

---

## Evaluation Metrics

### Precision

Precision measures how many of the retrieved documents were actually relevant.

```text
Precision = Relevant Retrieved Documents / Total Retrieved Documents
```

For example, if the system retrieves five documents and three are relevant:

```text
Precision = 3 / 5 = 60%
```

Higher precision means the search system is returning fewer irrelevant results.

### Recall

Recall measures how many of the documents considered relevant were successfully retrieved.

```text
Recall = Relevant Retrieved Documents / Total Relevant Documents
```

For example, if four documents are considered relevant and the system retrieves three of them:

```text
Recall = 3 / 4 = 75%
```

Higher recall means the search system is finding more of the information expected to be relevant.

---

## How the Evaluation Works

### 1. Create the ChromaDB collection

The project creates an ephemeral ChromaDB client and collection.

The documents are then inserted using their document IDs, content, and topic metadata.

### 2. Define the evaluation set

Each test query specifies which documents should be considered relevant.

For example:

```python
{
    "query": "What makes a Streamlit app persist?",
    "relevant_ids": ["doc_session"]
}
```

These expected IDs act as the ground truth for evaluating retrieval performance.

### 3. Query the semantic search system

Each query is sent to ChromaDB with a configurable number of results.

```python
results = eval_collection.query(
    query_texts=[query],
    n_results=n_results,
    include=["distances"]
)
```

ChromaDB returns the matching documents along with their distances.

For the distance metric used in this evaluation:

```text
Lower distance = greater similarity
```

### 4. Apply distance filtering

The evaluation function can optionally apply a distance threshold.

```python
if distance <= distance_threshold:
    retrieved_ids.append(id)
```

This allows the evaluation to test whether restricting results to closer matches improves precision.

### 5. Calculate relevance overlap

The expected relevant document IDs and retrieved document IDs are converted into sets.

Their intersection identifies the documents that were both expected and retrieved.

```python
overlap = len(relevant_ids & set(retrieved_ids))
```

This value is then used to calculate precision and recall.

### 6. Calculate averages

After evaluating all six queries, the system calculates average precision and recall across the evaluation set.

---

## Evaluation Configurations

The system evaluates search performance under three configurations:

### Configuration 1

```text
n_results = 3
```

The search system returns up to three results per query.

### Configuration 2

```text
n_results = 5
```

The search system returns up to five results per query.

### Configuration 3

```text
n_results = 5
distance_threshold = 1.4
```

The system retrieves up to five results but filters out documents whose distance exceeds the threshold.

Comparing these configurations demonstrates how retrieval parameters can affect search quality.

---

## Results

The program reports precision and recall for every query and calculates averages for each evaluation configuration.

The output follows this structure:

```text
=== Evaluation: n_results=3, threshold=None ===

Query 1: P=...% R=...%
Query 2: P=...% R=...%
Query 3: P=...% R=...%
...

AVERAGE: P=...% R=...%

=== Evaluation: n_results=5, threshold=None ===

...

=== Evaluation: n_results=5, threshold=1.4 ===

...
```

Because the evaluation is performed programmatically, the reported results are generated directly from the search system rather than manually entered.

---

## Analysis

The evaluation demonstrates several important characteristics of semantic search.

Queries containing distinctive concepts generally produced stronger retrieval results because the query had a closer semantic relationship to the relevant document.

Broader queries were more challenging. In some cases, the search system retrieved one relevant document while ranking another expected document lower. This demonstrates that semantic similarity does not always correspond perfectly with human-defined relevance.

Increasing `n_results` can provide the search system with more opportunities to retrieve relevant documents. However, returning additional documents can also introduce irrelevant results, which can reduce precision.

Distance filtering provides another way to control retrieval quality. A stricter threshold can remove weak matches and improve precision, but it can also remove relevant documents and reduce recall.

These results demonstrate the fundamental tradeoff between precision and recall when designing retrieval systems.

---

## Improving the Search System

The evaluation results suggest several potential improvements.

### Improve document content

Adding more context and stronger relationships between concepts could make documents easier for the embedding model to connect with relevant queries.

### Improve query design

Evaluation queries can be written to explicitly test the concepts they are intended to retrieve.

For example, a query about semantic search could explicitly mention both embeddings and cosine similarity when both concepts are expected to be retrieved.

### Experiment with chunking

Different chunk sizes and overlap settings could be evaluated to determine how document segmentation affects retrieval performance.

### Test additional thresholds

Evaluating multiple distance thresholds would make it possible to observe how precision and recall change as retrieval becomes more or less selective.

### Expand the evaluation dataset

A larger document collection and more evaluation queries would provide a stronger basis for measuring search performance.

### Compare embedding models

Different embedding models may produce different semantic relationships between queries and documents. Comparing models would provide another useful evaluation dimension.

---

## What This Project Demonstrates

This project demonstrates practical understanding of:

* Semantic search
* Vector databases
* ChromaDB
* Embedding-based retrieval
* Information-retrieval metrics
* Precision
* Recall
* Ground-truth evaluation sets
* Distance-based filtering
* Set operations in Python
* Parameter experimentation
* Retrieval-quality analysis

More importantly, the project demonstrates the idea that an AI search system should be **measured rather than assumed to work**.

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd Semantic-Search-Evaluation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install the dependency

```bash
pip install chromadb
```

### 4. Run the evaluation

```bash
python starter.py
```

The program automatically:

1. Creates the ChromaDB collection.
2. Loads the evaluation documents.
3. Runs all evaluation queries.
4. Calculates precision and recall.
5. Tests three retrieval configurations.
6. Calculates average performance.
7. Prints the evaluation analysis.

No external database or API key is required.

---

## Future Development

This evaluation framework could be extended to support:

* JSON or CSV evaluation datasets
* Automated threshold comparisons
* F1 score
* Mean Reciprocal Rank (MRR)
* Precision/recall curves
* Multiple embedding models
* Larger document collections
* Automated chunking experiments
* Retrieval-result visualization
* Automated evaluation reports

---

## Author

**Wayne Griffin**

Software engineering career changer focused on building practical applications involving APIs, data pipelines, semantic search, and AI/LLM technologies.
