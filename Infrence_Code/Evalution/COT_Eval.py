import json
import os
from openai import OpenAI
from tqdm import tqdm


QUESTION_BREAKDOWN_PROMPT = """
Refined prompt:
The following is the given question: {question}

Your task is to generate a detailed breakdown of the question in a tree-like structure. This breakdown should include all the given variables, objectives, and key concepts necessary to solve the question. You do not need to solve the question; just provide the breakdown.

The breakdown should include:
1. Question: question
2. Given Variables: List all variables provided in the question.
3. Objectives: List the main objectives that need to be achieved.
4. Key Concepts: Include the formulas, concepts, and theorems necessary to solve the question.

Example format:
- Question
  - Given Variables:
    - Variable 1
    - Variable 2
    - ...
  - Objectives:
    - Objective 1
    - Objective 2
    - ...
  - Key Concepts:
    - Concept 1: description, formulas, theorems
    - Concept 2: description, formulas, theorems
    - ...

"""


QUESTION_DECOMPOSITION_PROMPT = """
The following is a generated solution for the given question: {question}

Ground Truth Solution: {solution}

Generated Solution: {model_response}

Breakdown of the question: {breakdown}

Your task is to verify the following flags for the generated solution:
    1. Does the solution attempts to address the correct objective asked in the question?
    2. Are the correct values of the variables/entities from the question being used in the solution (in the applied formulas and reasoning)?

Important Notes:

    1. You don't have to verify whether the objective is solved correctly or not.
    2. You don't have to verify whether the reasoning performed, any formulas used, concepts used, or any calculations performed are correct or not.
    3. Focus only on whether the solution addresses the correct objective and uses the correct values from the question.

Examples:

    1. Objective Flag:
        If the question asks for the average speed of a car, check if the solution is focused on finding the average speed.
    2. Values Flag:
        If the question provides a distance of 150 miles and a time of 3 hours, ensure these exact values are used in the solution, and not
        some other value such as 120 miles in any step.
        
Return the flags in a list with 1 for correct and -1 for incorrect, e.g., [1, 1], [1, -1], [-1, 1], [-1, -1].

Output format:
```
Flag: [flag_1, flag_2]
```
"""



CONCEPT_PROMPT = """
The following is a generated solution to the given question {question}:
{model_response}

Ground Truth Solution : {solution}

The following is the detailed breakdown of the given question: {breakdown}

Your task is to:
Check each step in the generated solution against the relevant concepts and formulas provided in the detailed breakdown.
Verify whether the concepts and formulas are correctly used.

Your task is to verify the following flags for the generated solution:
        1. Is the concept is wrongly used in the solution. Check any instances where a concept has been misunderstood or incorrectly implemented.
        2. Is there a concept missing in the solution. Look for any important concepts or formulas from the detailed breakdown that are missing and should be present in the solution.

Just Return the flags in a list with 1 for correct and -1 for incorrect, e.g., [1, 1], [1, -1], [-1, 1], [-1, -1].

Output Format:
```
Concept Flag: [flag_1, flag_2]
```

"""


CALCULATION_PROMPT = """
The following is a generated solution of a given question {question}:
{model_response}


Ground Truth Solution : {solution}

Your task is to check for each step in the given solution and verify the mathematical calculations performed in the given solution.
You have to check all the operations and maths done within the application of the formuales. This includes all arithemtic operation, alegbraic manupilation,
application of mathematical procedures (integration, differentiation, etc.), handling of fractions, exponents, and radicals & numerical approximations or rounding.

Carefully review the mathematical calculations provided. Assign a flag based on the accuracy of the calculations. You will return the score with 1 for correct and -1 for incorrect
Example:
-1: There is at least one error in the mathematical calculations.
1: All the mathematical calculations are accurate and free from errors.


Note: You just have to check the mathematical operations such as addition, subtraction, division, multiplication, fraction, integeration, differentiation, power etc.
      You don't have to perform any kind of reasoning and solve the question. You are not the evaluator of whether the correct concept is used or whether the correct formulae is
      applied etc but you only have to evaluate if the mathematical operations are performed correctly. 

Just return in the output format as shown below.
Output Format:
```
Calculation flag: [flag]
```

"""


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_inference(model, prompt):
    prompt = prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful Physics Assistant Agent."},
            {"role": "user", "content": prompt}
        ]
    )
    response = response.choices[0].message.content.strip()
    return response


with open('question_dataset\OurDataset\Infrence\Llama3_70B\Physics_QA_Llama3_70B_cot_final.json', 'r') as file:
    data = json.load(file)

save_results = []
for idx, item in enumerate(tqdm(data, desc="Evaluating responses")):
    question = item['question'] + item['input']
    # print("question\n\n\n",question)
    # input = item['input']
    solution = item['solution']
    model_response = item['response']
    prompt = QUESTION_BREAKDOWN_PROMPT.format(question=question)
    # print(prompt)

    # Question BreakDown
    breakdown = run_inference("gpt-4o", prompt)
    # print(breakdown)

    # Question Decomposition
    prompt = QUESTION_DECOMPOSITION_PROMPT.format(question=question, model_response=model_response, solution=solution, breakdown=breakdown)
    # print(prompt)
    question_Decomposition_flags = run_inference("gpt-4o", prompt)
    print(question_Decomposition_flags)
    # divide the flagas and store in variables  Output in this format Flag: [flag_1, flag_2]
    flags_text = question_Decomposition_flags.strip().split("Flag: [")[1].split(']')[0].strip()
    correct_variable, final_objective = map(int, flags_text.split(','))
    # print("correct_variable\n\n\n",correct_variable)
    # print("final_objective\n\n\n",final_objective)

    #Concept Breakdown
    prompt = CONCEPT_PROMPT.format(question=question, model_response=model_response, solution=solution, breakdown=breakdown)
    # print(prompt)
    concept_score_flags = run_inference("gpt-4o", prompt)
    print(concept_score_flags)
    # divide the flags and store in variables  Output in this format Concept Flag: [flag_1, flag_2]
    flags_text = concept_score_flags.strip().split("Concept Flag: [")[1].split(']')[0].strip()
    concept_wrong_used, missing_concept = map(int, flags_text.split(','))
    # print("correct_wrong_used\n\n\n",concept_wrong_used)
    # print("missing_concept\n\n\n",missing_concept)

    #Calculation Error
    prompt = CALCULATION_PROMPT.format(question=question, model_response=model_response, solution=solution)
    # print(prompt)
    calculation_flags = run_inference("gpt-4o", prompt)
    print(calculation_flags)
    # divide the flags and store in variables  Output in this format Calculation flag: [flag]
    calculation_flag = int(calculation_flags.strip().split("Calculation flag: ")[1].split('\n')[0].strip())
    # print("calculation_flag\n\n\n",calculation_flag)

    # store in json in this format  
    flags={
        "Correct_Variable": correct_variable,
        "Final_Objective": final_objective,
        "Wrong_Concept": concept_wrong_used,
        "Concept_Missing": missing_concept,
        "Computation_Error": calculation_flag
    }

    
    item['flags'] = flags
    # print(item['flags'])

    save_results.append(item)

    if (idx+1) % 10 == 0:
        # Read existing results from the file, if it exists
        existing_results = []
        with open('physics_qa_Llama3_70B_flags.json', 'r') as file:
            existing_results = json.load(file)
        for i in save_results:
            existing_results.append(i)
        with open('physics_qa_Llama3_70B_flags.json', 'w') as file:
            json.dump(existing_results, file, indent=4)
        save_results = []
