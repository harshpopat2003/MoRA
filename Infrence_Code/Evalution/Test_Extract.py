import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract answers from model responses.")
    
    # Add arguments
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI API key")
    parser.add_argument('--model', type=str, default="gpt-4", help="Model to use for extraction")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the extracted answers JSON file")
    
    return parser.parse_args()

def run_inference(client, model, response):
    prompt = f"I am providing you a response from a model to a physics problem, termed 'Model Response'. You should extract the answer from the response. Directly output the extracted answer with no explanation. \n\nModel Response:{response}\n"
    # Run the model on the prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant for extracting the final answer from a solution to a problem."},
            {"role": "user", "content": prompt}
        ]
    )
    # Extract the response from the model
    extracted_answer = response.choices[0].message.content.strip()
    return extracted_answer

def main():
    # Parse arguments
    args = parse_arguments()

    # Set the API key
    os.environ["OPENAI_API_KEY"] = args.api_key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Initialize the OpenAI client

    # Load the input JSON file
    with open(args.input_file, 'r') as file:
        data = json.load(file)

    save_results = []
    
    # Iterate through each response in the JSON file
    for idx, item in enumerate(tqdm(data, desc="Evaluating responses")):
        response = item['response']

        # Evaluate the model's response
        extracted_answer = run_inference(client, args.model, response)
        print(extracted_answer)

        # Save the evaluation result
        item['extracted_answer'] = extracted_answer
        save_results.append(item)

        # Save the results every 10 responses
        if (idx + 1) % 10 == 0:
            # Read existing results from the file, if it exists
            existing_results = []
            if os.path.exists(args.output_file):
                with open(args.output_file, 'r') as file:
                    existing_results = json.load(file)

            # Append the new results to the existing results
            existing_results.extend(save_results)

            # Save the results to the file
            with open(args.output_file, 'w') as file:
                json.dump(existing_results, file, indent=4)
            
            save_results = []

    # Final save for remaining items
    if save_results:
        if os.path.exists(args.output_file):
            with open(args.output_file, 'r') as file:
                existing_results = json.load(file)
        else:
            existing_results = []
        
        # Append the remaining results
        existing_results.extend(save_results)
        
        with open(args.output_file, 'w') as file:
            json.dump(existing_results, file, indent=4)

if __name__ == "__main__":
    main()
