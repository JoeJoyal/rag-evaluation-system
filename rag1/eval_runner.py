from rag1.agents import ask
import json

with open("eval_dataset.jsonl", "r", encoding="utf-8") as file:
    test_cases = [json.loads(line) for line in file]

    results = []

    for test_cases in test_cases:

        question = test_cases["question"]
        expected_answer = test_cases["expected_answer"]

        print(f"\n Question: {question}")

        rag_result = ask(question)

        results ={
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": rag_result["answer"],
            "context": rag_result["context"]
        }

        results.append(results)
        print(f"Answer: {rag_result['answer']}")

with open("results_1.jsonl", "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result)+"\n")

print("\n Done! Results_1 saved to result_1.jsonl ")