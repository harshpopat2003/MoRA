import json

def evaluate_srcot_results(input_file):
    # Load the data from the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    total_questions = len(data)
    correct_answers = 0

    # Iterate through the data to evaluate answers
    for item in data:
        correct_answer = item.get("Correct Answer")
        srcot_answer = item.get("SRCoT Answer")

        # Check if the correct answer matches the SRCoT answer
        if correct_answer == srcot_answer:
            correct_answers += 1

    # Print the results
    print(f"Total Number of Questions: {total_questions}")
    print(f"Total Correct: {correct_answers}")

input_file = "../srcot_output/dynamic_phy_SRCoT_results_1.json"
evaluate_srcot_results(input_file)