import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate consistency of answers from infrence")
    
    # Add arguments
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI API key")
    parser.add_argument('--model', type=str, default="gpt-4o", help="Model to use for inference")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the final results JSON file")
    
    return parser.parse_args()

def run_inference(client, model, Question, Input, Standard_Answer, Model_Answer):
    # Run the model on the prompt
    prompt = f"""Below are two answers to a physics question.\n\n Question is {Question}\n\n Input Options:{Input}\n\n Standard Answer: {Standard_Answer} is the standard answer to the question, and \n\n Model Answer:{Model_Answer} is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
    Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
    If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just give the final jusgement in 1 or 0 nothing else.
    """
    
    # Run the model on the prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant evaluating the solution of a problem and comparing it to the ground truth."},
            {"role": "user", "content": prompt}
        ]
    )
    
    try:
        # Extract the response from the model
        response = int(response.choices[0].message.content.strip())
    except ValueError:
        response = 0
    return response

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
        Question = item['question']
        Input = item['input']
        Standard_Answer = item['answer']
        Model_Answer = item['extracted_answer']

        # Evaluate the model's response
        response = run_inference(client, args.model, Question, Input, Standard_Answer, Model_Answer)
        print(response)

        # Save the evaluation result
        item['judgement'] = response
        save_results.append(item)

        # Save the results every 10 responses
        if (idx + 1) % 10 == 0:
            # Read existing results from the file, if it exists
            existing_results = []
            if os.path.exists(args.output_file):
                with open(args.output_file, 'r') as file:
                    existing_results = json.load(file)
            
            # Append the new results to the existing results
            for i in save_results:
                existing_results.append(i)
            
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
