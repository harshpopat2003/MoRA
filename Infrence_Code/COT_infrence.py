import json
import os
import argparse
from together import Together
from tqdm import tqdm

# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on Dataset for CoT reasoning")
    
    # Add arguments
    parser.add_argument("--api_key", type=str, required=True, help="Your Together API key")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-70b-chat-hf", help="Model name to use for inference")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing the dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the inference results")
    
    return parser.parse_args()

# Define a function to run inference on the model
def run_inference(model, question, input, client):
    # Generating a prompt for the model
    prompt = f"Solve the following Question Step by Step \n\nQuestion: {question}\n\n Input: {input} \n\n finally respond with correctly input choice without fail."
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "assistant", "content": "You are an expert assistant in physics who can solve all the questions given to you correctly."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the response from the model
    response = response.choices[0].message.content.strip()
    return response

def main():
    # Parse arguments
    args = parse_args()
    
    # Set API key
    os.environ["TOGETHER_API_KEY"] = args.api_key
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))  # Initialize the Together client
    
    # Load the JSON file containing the questions and input
    with open(args.input_file, 'r') as file:
        data = json.load(file)
    
    # Initialize a list to save the results
    save_results = []
    
    # Iterate through each response in the JSON file
    for idx, item in enumerate(tqdm(data, desc="Evaluating responses")):
        question = item['question']
        input = item['input']

        # Evaluate the model's response
        response = run_inference(args.model, question, input, client)
        print(response)

        # Save the evaluation result
        item['response'] = response
        save_results.append(item)

        # Save the results every 10 responses
        if (idx+1) % 10 == 0:
            # Read existing results from the file, if it exists
            existing_results = []
            if os.path.exists(args.output_file):
                with open(args.output_file, 'r') as file:
                    existing_results = json.load(file)
            
            # Append the new results to the existing results
            existing_results.extend(save_results)
            
            # Save the results to a file
            with open(args.output_file, 'w') as file:
                json.dump(existing_results, file, indent=4)
            
            # Clear the save_results list
            save_results = []

    # Save any remaining results
    if save_results:
        with open(args.output_file, 'w') as file:
            json.dump(save_results, file, indent=4)

if __name__ == "__main__":
    main()
