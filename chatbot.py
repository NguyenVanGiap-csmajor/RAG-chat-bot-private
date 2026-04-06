from backend.rag import ask_rag


def main():
    while True:
        question = input("Enter your question (or 'exit' to quit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        result = ask_rag(question)
        print("\nAnswer:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")

        print()


if __name__ == "__main__":
    main()
