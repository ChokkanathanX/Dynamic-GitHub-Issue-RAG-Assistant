from SYNC import main as sync_database
from LANGRAPH_ROUTER import app


def main():

    print("\n")
    print("=" * 60)
    print("        DYNAMIC GITHUB RAG APPLICATION")
    print("=" * 60)

    # ------------------------------------------
    # Step 1: Synchronize knowledge base
    # ------------------------------------------

    print("\nSynchronizing knowledge base...")

    sync_database()

    # ------------------------------------------
    # Step 2: Start question answering
    # ------------------------------------------

    print("\n")
    print("=" * 60)
    print("        GITHUB RAG ASSISTANT")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question.strip():
            continue

        initial_state = {
            "question": question,
            "route": "",
            "issue_number": None,
            "state": None,
            "context": "",
            "answer": ""
        }

        result = app.invoke(initial_state)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(result["answer"])


if __name__ == "__main__":
    main()