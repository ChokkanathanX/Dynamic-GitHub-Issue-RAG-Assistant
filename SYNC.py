import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

BASE_URL = "https://api.github.com/repos/pallets/flask/issues"

HEADERS = {
    "Accept": "application/vnd.github+json"
}

TARGET_ISSUES = 500


# ==================================================
# STEP 1 — FETCH GITHUB ISSUES
# ==================================================

def fetch_issues():

    print("\n" + "=" * 60)
    print("STEP 1: FETCHING GITHUB ISSUES")
    print("=" * 60)

    all_issues = []

    page = 1

    while len(all_issues) < TARGET_ISSUES:

        params = {
            "state": "all",
            "per_page": 100,
            "page": page
        }

        response = requests.get(
            BASE_URL,
            headers=HEADERS,
            params=params
        )

        if response.status_code == 422:
            print("No more pages available.")
            break

        if response.status_code != 200:
            print("Failed:", response.status_code)
            break

        items = response.json()

        if not items:
            print("No more data available.")
            break

        # Remove pull requests
        actual_issues = [
            item
            for item in items
            if "pull_request" not in item
        ]

        all_issues.extend(actual_issues)

        print(
            f"Page {page}: "
            f"{len(items)} items → "
            f"{len(actual_issues)} issues"
        )

        # ------------------------------------------
        # Check GitHub pagination
        # ------------------------------------------

        if "next" not in response.links:
            print("Reached the last GitHub page.")
            break

        page += 1

    # Keep only target number
    all_issues = all_issues[:TARGET_ISSUES]

    print(
        f"\nTotal actual issues fetched: "
        f"{len(all_issues)}"
    )

    # Save raw data
    with open(
        "issues.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_issues,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("Saved to issues.json")

    return all_issues


# ==================================================
# STEP 2 — PROCESS GITHUB DATA
# ==================================================

def process_issues(issues):

    print("\n" + "=" * 60)
    print("STEP 2: PROCESSING ISSUES")
    print("=" * 60)

    processed_issues = []

    issue_count = 0
    pr_count = 0

    for item in issues:

        # Safety check
        if "pull_request" in item:

            pr_count += 1
            continue

        issue_count += 1

        labels = []

        for label in item["labels"]:
            labels.append(label["name"])

        cleaned_issue = {
            "id": item["id"],
            "number": item["number"],
            "title": item["title"],
            "body": item["body"] or "",
            "state": item["state"],
            "author": item["user"]["login"],
            "labels": labels,
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "url": item["html_url"]
        }

        processed_issues.append(cleaned_issue)

    print(f"Actual issues: {issue_count}")
    print(f"Pull requests: {pr_count}")

    # Save processed data
    with open(
        "processed_issues.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            processed_issues,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Processed {len(processed_issues)} issues"
    )

    print("Saved to processed_issues.json")

    return processed_issues


# ==================================================
# STEP 3 — CONNECT TO CHROMADB
# ==================================================

def connect_chroma():

    print("\n" + "=" * 60)
    print("STEP 3: CONNECTING TO CHROMADB")
    print("=" * 60)

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="github_issues"
    )

    print(
        f"Existing issues in ChromaDB: "
        f"{collection.count()}"
    )

    return collection


# ==================================================
# STEP 4 — DETECT CHANGES
# ==================================================

def detect_changes(
    processed_issues,
    collection
):

    print("\n" + "=" * 60)
    print("STEP 4: DETECTING CHANGES")
    print("=" * 60)

    existing = collection.get()

    existing_metadata = existing["metadatas"]

    existing_issues = {}

    for metadata in existing_metadata:

        issue_number = metadata["number"]

        existing_issues[issue_number] = metadata

    new_issues = []
    updated_issues = []
    unchanged_issues = []

    for issue in processed_issues:

        issue_number = issue["number"]

        # ------------------------------------------
        # NEW ISSUE
        # ------------------------------------------

        if issue_number not in existing_issues:

            new_issues.append(issue)

        # ------------------------------------------
        # EXISTING ISSUE
        # ------------------------------------------

        else:

            existing_issue = existing_issues[
                issue_number
            ]

            if (
                issue["updated_at"]
                != existing_issue["updated_at"]
            ):

                updated_issues.append(issue)

            else:

                unchanged_issues.append(issue)

    print(f"New issues:       {len(new_issues)}")
    print(f"Updated issues:   {len(updated_issues)}")
    print(f"Unchanged issues: {len(unchanged_issues)}")

    return (
        new_issues,
        updated_issues,
        unchanged_issues
    )


# ==================================================
# STEP 5 — EMBED + UPDATE CHROMADB
# ==================================================

def update_chroma(
    new_issues,
    updated_issues,
    collection
):

    print("\n" + "=" * 60)
    print("STEP 5: UPDATING CHROMADB")
    print("=" * 60)

    issues_to_process = (
        new_issues + updated_issues
    )

    if not issues_to_process:

        print("No new or updated issues.")

        print(
            "ChromaDB is already up to date."
        )

        return

    print(
        f"Generating embeddings for "
        f"{len(issues_to_process)} issues..."
    )

    # Load embedding model
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    documents = []
    ids = []
    metadatas = []

    for issue in issues_to_process:

        document = f"""
Title: {issue["title"]}

Description:
{issue["body"]}

Labels:
{", ".join(issue["labels"])}
"""

        documents.append(document)

        ids.append(str(issue["id"]))

        metadatas.append({
            "issue_id": str(issue["id"]),
            "number": issue["number"],
            "state": issue["state"],
            "author": issue["author"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "url": issue["url"]
        })

    # Generate embeddings
    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    # Add new + update existing
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Added/updated: "
        f"{len(issues_to_process)}"
    )

    print(
        f"Total in ChromaDB: "
        f"{collection.count()}"
    )


# ==================================================
# MAIN SYNC PIPELINE
# ==================================================

def main():

    print("\n")
    print("#" * 60)
    print("        DYNAMIC GITHUB RAG SYNC")
    print("#" * 60)

    # 1. Fetch
    issues = fetch_issues()

    # 2. Process
    processed_issues = process_issues(
        issues
    )

    # 3. Connect ChromaDB
    collection = connect_chroma()

    # 4. Detect changes
    (
        new_issues,
        updated_issues,
        unchanged_issues
    ) = detect_changes(
        processed_issues,
        collection
    )

    # 5. Update ChromaDB
    update_chroma(
        new_issues,
        updated_issues,
        collection
    )

    print("\n" + "#" * 60)
    print("              SYNC COMPLETE")
    print("#" * 60)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()