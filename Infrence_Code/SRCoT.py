import json
import os
from together import Together
from tqdm import tqdm

client = Together(api_key="KEY")  # Initialize the Together client

# Function to run inference with dynamic step-by-step verification (SRCoT)
def run_inference_srot(model, question):
    initial_prompt = f"You are an expert in solving physics problems. Solve the following question step by step using detailed reasoning.\n\nQuestion: {question}\n\n Provide the reasoning for each step before proceeding to the next. Indicate when the solution is complete and choose the final answer (A, B, C, or D)."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": initial_prompt}
        ]
    )

    final_response = response.choices[0].message.content.strip()

    step = 1
    final_answer = None

    while True:
        # Check if any of the options A, B, C, or D are already in the final_response
        for option in ["A", "B", "C", "D"]:
            if f" {option}" in final_response or f"ANSWER-{option}" in final_response:
                final_answer = option
                break

        # If an answer has been found, append the "ANSWER-X" format and break the loop
        if final_answer:
            final_response += f"\nANSWER-{final_answer}"
            break

        # Otherwise, ask for the next step
        next_step_prompt = f"Step {step}: {final_response}\n\n Please explain the next step of reasoning carefully."
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": next_step_prompt}
            ]
        )
        step_output = response.choices[0].message.content.strip()

        # Append the new step to the response
        final_response += f"\nStep {step}: {step_output}\n"
        step += 1

    return final_response, final_answer

# Helper function to extract the answer letter from the SRCoT response
def extract_answer_from_response(response):
    for option in ["A", "B", "C", "D"]:
        if f"ANSWER-{option}" in response:
            return option
    return None  # If no valid option is found, return None

# Main function to process the data and save the results
def process_data_and_save_results(input_file, output_file, model_name):
    with open(input_file, 'r') as file:
        data = json.load(file)

    save_results = []
    correct_count = 0

    print("Data loaded")
    total = len(data)

    # Iterate through each response in the JSON file
    for idx, item in enumerate(tqdm(data, desc="Evaluating responses")):
        question = item['question']
        correct_answer = item['answer'][0]  # Extract the correct answer from the input

        # Run inference using SRCoT (with dynamic step reconsideration)
        response, srcot_answer = run_inference_srot(model_name, question)

        # Extract the SRCoT answer letter from the model's response
        predicted_answer = extract_answer_from_response(response)

        # Prepare the result object
        result = {
            "Input Question": question,
            "Correct Answer": correct_answer,
            "SRCoT Answer": predicted_answer
        }
        save_results.append(result)

        # Check if the prediction matches the correct answer
        if predicted_answer == correct_answer:
            correct_count += 1

        # Save the results every 10 responses
        if (idx + 1) % 10 == 0:
            existing_results = []
            try:
                with open(output_file, 'r') as file:
                    existing_results = json.load(file)
            except FileNotFoundError:
                existing_results = []
            existing_results.extend(save_results)
            with open(output_file, 'w') as file:
                json.dump(existing_results, file, indent=4)
            save_results = []

    # Final save of remaining results
    with open(output_file, 'w') as file:
        json.dump(save_results, file, indent=4)

    # Calculate and print accuracy
    accuracy = correct_count / total
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total})")

# Define the input and output files and run the process
input_file = '../../Data/dynamic_phy.json'
output_file = '../srcot_output/dynamic_phy_SRCoT_results_3.json'
model_name = "google/gemma-2-27b-it"

process_data_and_save_results(input_file, output_file, model_name)