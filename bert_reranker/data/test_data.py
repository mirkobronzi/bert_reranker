import json

from tqdm import tqdm

def evaluate_model(ret_trainee, qa_pairs_json_file):

    # Run the test on samples from natq to sanity check evreything is correct
    #  qa_pairs =
    with open(qa_pairs_json_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    correct = 0
    for question, answers in tqdm(qa_pairs):

        out = ret_trainee.retriever.predict(question, answers)

        if out[2][0] == 0:  # answers[0] is always the correct answer
            correct += 1

    acc = correct / len(qa_pairs) * 100
    print("Accuracy: %", acc)
