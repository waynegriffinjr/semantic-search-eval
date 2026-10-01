"""
L7 — Evaluate Your Search System  (STARTER)
=============================================
Run with:
    python starter.py

Your goal: build an evaluation framework that measures precision and recall
for a semantic search system across multiple settings.

Required features:
    1. A ChromaDB collection with at least 12 documents across 3–4 topics
    2. An evaluation set: at least 6 test queries, each with a list of
       expected relevant document IDs
    3. An evaluate() function that computes per-query and average
       precision & recall
    4. Results at 3 different settings (vary threshold, n_results, or both)
    5. A written analysis in the output

Key concepts:
    Precision — of the results returned, how many were relevant?
        precision = len(relevant ∩ retrieved) / len(retrieved)

    Recall — of all relevant documents, how many did we retrieve?
        recall = len(relevant ∩ retrieved) / len(relevant)

    ChromaDB returns distances (lower = more similar).
    Convert to similarity for thresholding:
        similarity = 1 - distance   (for cosine distance)
        OR just filter by distance <= threshold

    Example evaluation set entry:
        {
            "query": "How does FastAPI validate request data?",
            "relevant_ids": ["doc_03", "doc_07"]
        }
"""

import chromadb

# ── Documents ─────────────────────────────────────────────────────────────────

documents = {
    "doc_jwt": "JWT tokens provide stateless authentication for REST APIs. The server creates a signed token containing user info.",
    "doc_pydantic": "Pydantic models define data schemas for FastAPI. They automatically validate incoming request data.",
    "doc_session": "Streamlit session state persists data across re-runs. Initialize with: if key not in st.session_state.",
    "doc_cors": "CORS middleware in FastAPI allows cross-origin requests from frontend applications running on different ports.",
    "doc_embed": "Embeddings convert text into numerical vectors capturing semantic meaning. Similar texts get similar vectors.",
    "doc_cosine": "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical direction.",
    "doc_chunk": "Chunking splits documents into smaller pieces for embedding. Chunk size affects search precision and recall.",
    "doc_chroma": "ChromaDB is a vector database for storing and querying embeddings. It supports metadata filtering.",
    "doc_flex": "CSS Flexbox arranges elements in rows or columns. Use display:flex on the container.",
    "doc_dom": "The DOM is the browser's tree representation of HTML. JavaScript uses it to modify page content.",
    "doc_injection": "What Is Prompt Injection? Prompt injection is when a user (or a piece of data the model processes) includes instructions that override or modify the AIs intended behavior. It is the AI equivalent of SQL injection - untrusted input manipulating the systems instructions. The fundamental problem: LLMs cannot reliably distinguish between instructions from the developer and instructions from the user. Both look like text. The model processes all text the same way.",
    "doc_mitigations": "1. Defense in Depth, 2. Input Validation, 3. Output Validation, 4. Least Privilege, 5. Separate Instruction and Data Channels",
}

ids = list(documents.keys())

metadatas = [
    {"topic": "JWT"},
    {"topic": "Pydantic"},
    {"topic": "Streamlit"},
    {"topic": "CORS"},
    {"topic": "Embedding"},
    {"topic": "Cosine Similarity"},
    {"topic": "Chunking"},
    {"topic": "ChromaDB"},
    {"topic": "CSS Flexbox"},
    {"topic": "JavaScript DOM"},
    {"topic": "Prompt Injection"},
    {"topic": "Injection Defense"},
]

# ── ChromaDB setup ────────────────────────────────────────────────────────────

eval_client = chromadb.EphemeralClient()
eval_collection = eval_client.get_or_create_collection(
    name="my_eval",
    metadata={"description": "AI Engineering Documents"},
)

eval_collection.upsert(
    documents=list(documents.values()),
    metadatas=metadatas,
    ids=ids
)

# ── Evaluation set ────────────────────────────────────────────────────────────

eval_set = [
    {
        "query": "What is the best way to protect against prompt injection?",
        "relevant_ids": ["doc_injection", "doc_mitigations"]
    },
    {
        "query": "How are API requests authenticated?",
        "relevant_ids": ["doc_jwt", "doc_cors"]
    },
    {
        "query": "What makes a Streamlit app persist?",
        "relevant_ids": ["doc_session"]
    },
    {
        "query": "How do semantic searches work in AI Engineering?",
        "relevant_ids": ["doc_embed", "doc_cosine"]
    },
    {
        "query": "How should I prepare documents for a search system?",
        "relevant_ids": ["doc_chunk", "doc_chroma"]
    },
    {
        "query": "How do static webpages become interactive?",
        "relevant_ids": ["doc_dom", "doc_flex"]
    },
]



# ── Evaluation function ────────────────────────────────────────────────────────
def evaluate(n_results: int, distance_threshold: float = None):
    """
    Run every query in eval_set and measure how well our semantic search
    system retrieves the documents we consider relevant.

    We calculate two common information-retrieval metrics:

    - Precision: Of the documents we retrieved, how many were relevant?
    - Recall: Of the documents that were relevant, how many did we retrieve?

    Args:
        n_results:
            The maximum number of documents ChromaDB should return
            for each query.

        distance_threshold:
            Optional cutoff for similarity distance.

            If provided, we only keep results where the distance is
            less than or equal to this value.

            Lower distance generally means the result is more similar
            to the query when using a distance-based metric.

    Returns:
        A tuple containing:
            (average_precision, average_recall)
    """
    
    # Print the settings being used for this evaluation.
    print(
        f"\n=== Evaluation: "
        f"n_results={n_results}, "
        f"threshold={distance_threshold} ==="
    )

    # These lists will store the precision and recall scores
    # from each individual query.
    # Used later for calculating averages
    precisions = []
    recalls    = []
    
    

    for i, entry in enumerate(eval_set, 1):
        query        = entry["query"]
        relevant_ids = set(entry["relevant_ids"])
        
        results = eval_collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances"]
        )

        retrieved_ids = []
        
        for id, distance in zip(
            results["ids"][0],
            results["distances"][0]
        ):
            if distance_threshold is None:
                retrieved_ids.append(id)
            else:
                if distance <= distance_threshold:
                    retrieved_ids.append(id)
                
                
                    
 
        overlap   = len(
            relevant_ids & set(retrieved_ids) # IDs that exist in both sets (&)
        )
        
        precision = (
            overlap / len(retrieved_ids)
            if retrieved_ids 
            else 0
        )
        
        recall = (
            overlap / len(relevant_ids)
            if relevant_ids
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)

        print(f"  Query {i}: "
              f"P={precision*100:.1f}% "
              f"R={recall*100:.1f}%  "
              f"| '{query[:50]}'"
        )
        


    # Once every query has been evaluated, calculate the average
    # precision across all queries while accounting for zero returns
    avg_p = (
        sum(precisions) / len(precisions)
        if precisions
        else 0
    )
    
    avg_r = (
        sum(recalls) / len(recalls)   
        if recalls
        else 0
    )
    
    # Print the overall evaluation results.
    print(
        f"  AVERAGE: "
        f"P={avg_p*100:.1f}%  "
        f"R={avg_r*100:.1f}%"
    )
    
    return avg_p, avg_r


# ── Run at 3 settings ─────────────────────────────────────────────────────────

evaluate(n_results=3)

evaluate(n_results=5)

evaluate(n_results=5, distance_threshold=1.4)




# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n=== ANALYSIS ===")

print(f"\n{'=' * 60}")
print(f"AVERAGE Precision: {avg_precision:.1%}")
print(f"AVERAGE Recall:    {avg_recall:.1%}")


print("=== DEVELOPER ANALYSIS ===")

print("""
      
The queries that performed best were Queries 1, 2, 3, and 5 because they consistently achieved 100 percent recall in the evaluations without a threshold. This means the search system found every document marked as relevant for those queries. Query 3, about Streamlit persistence, was especially successful because the relevant document was distinctive and closely matched the query. Query 1, about prompt injection, and Query 5, about preparing documents for search, also matched their related documents effectively.

Queries 4 and 6 were the weakest. Both achieved only 50 percent recall in every setting, meaning the system found only one of the two expected relevant documents. Query 4 asked about semantic search and may have retrieved the embedding document but missed the cosine similarity document. Query 6 asked how webpages become interactive and may have found the DOM document but missed the Flexbox document. These results suggest that the queries or document descriptions did not connect strongly enough to both relevant topics in the embedding search.

To improve the weakest queries, I would make the query wording more specific and include keywords from both expected documents. For example, Query 4 could mention “embeddings and cosine similarity,” while Query 6 could mention “the DOM, JavaScript, and CSS Flexbox.” I could also rewrite the document text to describe its relationship to the broader topic more clearly, or add additional documents with more detailed explanations. Increasing `n_results` alone did not improve recall for these queries, so the main problem is ranking or semantic similarity rather than returning too few results.

Increasing `n_results` from 3 to 5 reduced precision from 50.0% to 30.0% while leaving average recall unchanged at 83.3%. The additional results were often irrelevant, so precision decreased. Recall did not improve because the extra results did not include the relevant documents that had already been missed. This demonstrates the usual tradeoff: retrieving more documents can increase recall when relevant documents appear lower in the ranking, but it can also lower precision by adding unrelated results.

The distance threshold of `1.4` produced the highest precision, 94.4%, but reduced recall to 75.0%. The threshold removed many irrelevant results, which improved precision, but it also removed some relevant documents. Therefore, the threshold setting was highly selective and produced cleaner results at the cost of missing some information.      
   
""")
#       Answer:
#         - Which queries consistently performed well? Why?
#         - Which queries failed? What caused the low score?
#         - What would you change to improve the weakest queries?
#         - How did increasing n_results affect precision vs recall?
