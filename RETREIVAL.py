import requests
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Configuration
# ==========================================

OWNER = "pallets"
REPO = "flask"


# ==========================================
# 2. Load embedding model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 3. Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="github_issues"
)


# ==========================================
# 4. Semantic Search
# ==========================================

def semantic_search(question, n_results=3, state=None):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    if state:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"state": state}
        )

    else:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

    return results


# ==========================================
# 5. Exact Issue Lookup
# ==========================================

def exact_issue(issue_number):

    result = collection.get(
        where={
            "number": issue_number
        }
    )

    return result


# ==========================================
# 6. Latest Issues from GitHub API
# ==========================================

def latest_issues(limit=5):

    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/issues"
    )

    params = {
        "state": "all",
        "sort": "updated",
        "direction": "desc",
        "per_page": 30
    }

    response = requests.get(
        url,
        params=params
    )

    if response.status_code != 200:
        raise Exception(
            f"GitHub API error: {response.status_code}"
        )

    issues = response.json()

    actual_issues = []

    for issue in issues:

        # GitHub's /issues endpoint also returns PRs
        if "pull_request" not in issue:
            actual_issues.append(issue)

        if len(actual_issues) == limit:
            break

    return actual_issues
# ==========================================
# 7. Hybrid Search
# ==========================================
def hybrid_search(question, recent_limit=30, top_k=5,state=None):

    # Get recent issues from GitHub
    recent_issues = latest_issues(recent_limit)
    if state:
        recent_issues = [
            issue
            for issue in recent_issues
            if issue["state"] == state
        ]
    if not recent_issues:
        return []

    # Create text representation of each issue
    issue_texts = []

    for issue in recent_issues:

        text = f"""
        Title: {issue["title"]}

        Description:
        {issue["body"] or ""}
        """

        issue_texts.append(text)

    # Embed the user's question
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    # Embed recent issues
    issue_embeddings = embedding_model.encode(
        issue_texts,
        normalize_embeddings=True
    )

    # Cosine similarity
    similarities = np.dot(
        issue_embeddings,
        question_embedding
    )

    # Sort by similarity
    ranked_indices = np.argsort(
        similarities
    )[::-1]

    # Select top K
    results = []

    for index in ranked_indices[:top_k]:

        issue = recent_issues[index]

        results.append({
            "issue": issue,
            "similarity": float(similarities[index])
        })

    return results

# ==========================================
# 8. Main
# ==========================================

def main():

    question = input("\nAsk a question: ")

    print("\nYour question:")
    print(question)

    print("\nRetrieval functions are ready.")
    print("The router will decide which one to use.")


# ==========================================
# 9. Program entry point
# ==========================================

if __name__ == "__main__":
    main()