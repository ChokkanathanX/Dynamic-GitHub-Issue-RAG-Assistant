from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from LLM_LOOKUP import llm_route

from RETREIVAL import (
    semantic_search,
    exact_issue,
    latest_issues,
    hybrid_search
)

from CONTEXT_CONNECTOR import (
    context_from_semantic,
    context_from_exact,
    context_from_latest
)

from GENERATION import generate_answer


# ============================================================
# GRAPH STATE
# ============================================================

class GraphState(TypedDict):

    question: str

    # Information extracted by the LLM router
    route: str
    issue_number: int | None
    state: str | None

    # Data passed between nodes
    context: str

    # Final response
    answer: str


# ============================================================
# ROUTER NODE
# ============================================================

def router_node(state: GraphState):

    print("\nRunning LLM router...")

    intent = llm_route(
        state["question"]
    )

    print("Selected retrieval:", intent["route"])
    print("Issue number:", intent["issue_number"])
    print("State:", intent["state"])

    return {
        "route": intent["route"],
        "issue_number": intent["issue_number"],
        "state": intent["state"]
    }


# ============================================================
# EXACT RETRIEVAL NODE
# ============================================================

def exact_node(state: GraphState):

    print("\nRunning exact retrieval...")

    result = exact_issue(
        state["issue_number"]
    )

    context = context_from_exact(
        result
    )

    return {
        "context": context
    }


# ============================================================
# SEMANTIC RETRIEVAL NODE
# ============================================================

def semantic_node(state: GraphState):

    print("\nRunning semantic retrieval...")

    results = semantic_search(
        state["question"],
        n_results=3,
        state=state["state"]
    )

    context = context_from_semantic(
        results
    )

    return {
        "context": context
    }


# ============================================================
# LATEST RETRIEVAL NODE
# ============================================================

def latest_node(state: GraphState):

    print("\nRunning latest retrieval...")

    issues = latest_issues(
        5
    )

    context = context_from_latest(
        issues
    )

    return {
        "context": context
    }


# ============================================================
# HYBRID RETRIEVAL NODE
# ============================================================

def hybrid_node(state: GraphState):

    print("\nRunning hybrid retrieval...")

    results = hybrid_search(
        state["question"],
        recent_limit=30,
        top_k=5,
        state=state["state"]
    )

    context = ""

    for result in results:

        issue = result["issue"]
        similarity = result["similarity"]

        context += f"""
Issue #{issue["number"]}
Title: {issue["title"]}
State: {issue["state"]}
Author: {issue["user"]["login"]}
Created: {issue["created_at"]}
Updated: {issue["updated_at"]}
URL: {issue["html_url"]}
Similarity: {similarity:.3f}

Description:
{issue["body"]}

----------------------------------------
"""

    return {
        "context": context
    }


# ============================================================
# GENERATION NODE
# ============================================================

def generation_node(state: GraphState):

    print("\nGenerating answer...")

    answer = generate_answer(
        state["question"],
        state["context"]
    )

    return {
        "answer": answer
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def choose_retrieval(state: GraphState):

    return state["route"]


# ============================================================
# BUILD GRAPH
# ============================================================

graph = StateGraph(
    GraphState
)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

graph.add_node(
    "router",
    router_node
)

graph.add_node(
    "exact",
    exact_node
)

graph.add_node(
    "semantic",
    semantic_node
)

graph.add_node(
    "latest",
    latest_node
)

graph.add_node(
    "hybrid",
    hybrid_node
)

graph.add_node(
    "generation",
    generation_node
)


# ------------------------------------------------------------
# START → ROUTER
# ------------------------------------------------------------

graph.add_edge(
    START,
    "router"
)


# ------------------------------------------------------------
# ROUTER → RETRIEVAL
# ------------------------------------------------------------

graph.add_conditional_edges(
    "router",
    choose_retrieval,
    {
        "exact": "exact",
        "semantic": "semantic",
        "latest": "latest",
        "hybrid": "hybrid"
    }
)


# ------------------------------------------------------------
# RETRIEVAL → GENERATION
# ------------------------------------------------------------

graph.add_edge(
    "exact",
    "generation"
)

graph.add_edge(
    "semantic",
    "generation"
)

graph.add_edge(
    "latest",
    "generation"
)

graph.add_edge(
    "hybrid",
    "generation"
)


# ------------------------------------------------------------
# GENERATION → END
# ------------------------------------------------------------

graph.add_edge(
    "generation",
    END
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    question = input("\nAsk a question: ")

    initial_state = {
        "question": question,
        "route": "",
        "issue_number": None,
        "state": None,
        "context": "",
        "answer": ""
    }

    result = app.invoke(
        initial_state
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )