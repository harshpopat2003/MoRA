import json
import os
import argparse
from together import Together
from tqdm import tqdm

from promptshhh import Heat_Transfer, Waves_on_String, Alternating_Current, Capacitor, Communication_System, Elasticity, Electromagnetic_Waves, Kinematics_2D, Current_Electricity, Kinematics_1D, Kinetic_Theory_of_Gases, Magnetism, Nuclear_Physics, Thermodynamics, Work_Power_Energy, Centre_of_Mass, Electrostatics, Radioactivity, Ray_Optics, Rotational_Motion, Semiconductors, Simple_Harmonic_Motion, Sound_Waves, Wave_Optics, Electromagnetic_Induction, Gravitation, Fluid_Mechanics, Friction, Thermal_Expansion

# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run few-shot inference on dataset")

    # Add arguments for API key, model, input file, output file
    parser.add_argument("--api_key", type=str, required=True, help="Your Together API key")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-70b-chat-hf", help="Model name to use for inference")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing the dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the inference results")

    return parser.parse_args()

# Define the function to create the prompt
def create_prompt(few_shot_prompt, question, input):
    prompt = f"Solve the following Question and Input given correctly giving the complete solution\n\nQuestion: {question}\n\n Input: {input} \n\n finally respond along with correctly input choice without fail.\n\n\n Given Below are some questions related to the chapter along with their solutions.\n\n{few_shot_prompt}"
    return prompt

# Define the function to run inference
def run_inference(model, prompt, client):
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

    # Initialize the dictionary with chapters and corresponding few-shot prompts
    chapters_dict = {
        "Heat Transfer": Heat_Transfer,
        "Waves on String": Waves_on_String,
        "Alternating Current": Alternating_Current,
        "Capacitor": Capacitor,
        "Communication System": Communication_System,
        "Elasticity": Elasticity,
        "Electromagnetic Waves": Electromagnetic_Waves,
        "Kinematics 2D": Kinematics_2D,
        "Current Electricity": Current_Electricity,
        "Kinematics 1D": Kinematics_1D,
        "Kinetic Theory of Gases": Kinetic_Theory_of_Gases,
        "Magnetism": Magnetism,
        "Nuclear Physics": Nuclear_Physics,
        "Thermodynamics": Thermodynamics,
        "Work Power Energy": Work_Power_Energy,
        "Centre of Mass": Centre_of_Mass,
        "Electrostatics": Electrostatics,
        "Radioactivity": Radioactivity,
        "Ray Optics": Ray_Optics,
        "Rotational Motion": Rotational_Motion,
        "Semiconductors": Semiconductors,
        "Simple Harmonic Motion": Simple_Harmonic_Motion,
        "Sound Waves": Sound_Waves,
        "Wave Optics": Wave_Optics,
        "Electromagnetic Induction": Electromagnetic_Induction,
        "Gravitation": Gravitation,
        "Fluid Mechanics": Fluid_Mechanics,
        "Friction": Friction,
        "Thermal Expansion": Thermal_Expansion
    }

    # Load the JSON file containing the questions and inputs
    with open(args.input_file, 'r') as file:
        data = json.load(file)

    # Initialize the list to save the results
    save_results = []

    # Iterate through each response in the JSON file
    for idx, item in enumerate(tqdm(data, desc="Evaluating responses")):
        question = item['question']
        input_data = item['input']
        chapter = item['chapter']

        if chapter in chapters_dict:
            # Get the few-shot prompt for the chapter
            few_shot_prompt = chapters_dict[chapter]

            # Create the prompt
            prompt = create_prompt(few_shot_prompt, question, input_data)

            # Evaluate the model's response
            response = run_inference(args.model, prompt, client)
            print(response)

            # Save the evaluation result
            item['response'] = response
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

                # Clear the save_results list
                save_results = []
        else:
            # Print the chapter not found
            print(f"Chapter not found: {chapter}")
            continue

    # Save any remaining results at the end
    if save_results:
        with open(args.output_file, 'w') as file:
            json.dump(save_results, file, indent=4)

if __name__ == "__main__":
    main()
