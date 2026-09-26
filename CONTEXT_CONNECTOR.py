# CONTEXT.py


# ==========================================
# 1. Build context from semantic search
# ==========================================

def context_from_semantic(results):

    context = ""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i in range(len(documents)):

        metadata = metadatas[i]
        document = documents[i]

        context += f"""
Issue #{metadata["number"]}
State: {metadata["state"]}
Author: {metadata["author"]}
URL: {metadata["url"]}

Content:
{document}

----------------------------------------
"""

    return context


# ==========================================
# 2. Build context from exact issue lookup
# ==========================================

def context_from_exact(result):

    if not result["documents"]:
        return ""

    context = ""

    for i in range(len(result["documents"])):

        metadata = result["metadatas"][i]
        document = result["documents"][i]

        context += f"""
Issue #{metadata["number"]}
State: {metadata["state"]}
Author: {metadata["author"]}
URL: {metadata["url"]}

Content:
{document}

----------------------------------------
"""

    return context


# ==========================================
# 3. Build context from GitHub API
# ==========================================

def context_from_latest(issues):

    context = ""

    for issue in issues:

        context += f"""
Issue #{issue["number"]}
Title: {issue["title"]}
State: {issue["state"]}
Author: {issue["user"]["login"]}
Created: {issue["created_at"]}
Updated: {issue["updated_at"]}
URL: {issue["html_url"]}

Description:
{issue["body"]}

----------------------------------------
"""

    return context