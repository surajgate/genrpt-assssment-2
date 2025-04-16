from agent import answer_question

if __name__ == "__main__":
    # Example usage and testing
    questions: list[str] = [
        # SQL-Oriented Questions (Structured Data)
        "List all matches played by Chennai Super Kings in 2023",
        "Show the top 5 batsmen with highest strike rates in IPL 2023",
        "Which team won the most matches in 2022?",

        # PDF-Oriented Questions (Unstructured Text)
        "What is the home ground of Punjab Kings?",
        "Tell me about MS Dhoni's captaincy style",
        "What are the key features of IPL's impact player rule?",
    ]

    for question in questions:
        print(f"\nOriginal Question: {question}")
        result = answer_question(question)
        print(f"Final Question: {result['final_question']}")
        print(f"Answer: {result['answer']}")
        print(
            f"Source: {result['source']} (confidence: {result['confidence']:.2f})")
        print(f"Rephrased: {result['was_rephrased']}")
        if result.get("sql_query"):
            print(f"SQL Query: {result['sql_query']}")
        print("=" * 80)
