# ACF - Adaptive Correction Framework 
## Requirements

The code is written in Python 3.10. Before running, you need to first install the required packages by typing following commands (Using a virtual environment is recommended):

```
conda create --name ACF python==3.10
conda activate ACF
pip3 install -r requirements.txt
```

## Datasets
You can access the datasets in `ACF/Dataset`

**Datasets:**

1. `MMLU_College_Physics.json`
2. `MMLU_High_School_Physics.json`
4. `SciEval_Static_Physics.json`

## Agent

ACF implementation can be found in `ACF/agent`

**Python Files:**

1. `agent.py`: Contains the `ACF` agent class
2. `prompts.py`: Contains all the error identification & refinement prompts
3. `utils.py`: Contains utilities functions for using GraphRAG
4. `main.py`: Contains `evaluate()` function for running ACF on datasets

## Knowledge Base

Download the `GRAPH_RAG` folder which contains Topic wise KBs from here: [Google Drive](https://drive.google.com/file/d/1reSQgvrqGwh_lNEXLbJlbaRCHIYnmDLd/view)

Replace empty  `ACF/GRAPH_RAG` folder with the downloaded one.

Using GraphRAG, local search is performed on these KBs to obtain conceptual contexts.

## Experiments

### Run ACF

**Steps:**

1. Add your `TOGETHER_API_KEY` &  `OPENAI_API_KEY` in `main.py`

2. Add your `GRAPHRAG_API_KEY` in `utils.py`, it's same as your `OPENAI_API_KEY`

3. Add `model` in `main.py` as base LLM:

   1. `Llama-3-70B`: "meta-llama/Llama-3-70b-chat-hf"
   2. `Gemma-2-27B`: " google/gemma-2-27b-it"

   You can select any other LLM as well, refer to for model path: [TogetherAI](https://docs.together.ai/docs/chat-models)

4. Add `llm_model` in `main.py` as error identifier, for our experiments we used: `gpt-4o` 

5.  Run `main.py` with following args:

   1. `dataset_filename`: The name of the dataset file to test, example `PhysicsQA.json` , make sure dataset contains `response` for each questions that needs to be refined. Run `CoT` inference first on the dataset (see the next section) and use that filename and path.
   2. `max_steps`: The maximum number of iteration steps for refinement
   3. `graph_rag_dir`: Directory path for GRAPH_RAG data
   4. `dataset_dir`: Directory path for datasets, should include `response` for each question that needs to be refined. 
   5. `result_dir`: Directory path for saving results

   Example: 

   ```
   python main.py PhysicsQA.json --max_steps 5 --graph_rag_dir 'ACF/GRAPH_RAG' --dataset_dir 'ACF/Dataset' --result_dir 'ACF/Results'
   ```

   

### Run Baselines Infrences 
Scripts for running inference on any dataset using the following methods:
- **AO (Answer Only)**: The model generates only the final answer to the given question.
- **COT (Chain of Thought)**: The model solves the question step by step and explains its reasoning process.
- **Few-Shot**: The model uses a few-shot learning approach, where it is provided with examples from the same chapter to aid in solving the target question.

Install the necessary libraries using the following command:

```bash
pip install together tqdm argparse
```

#### How to Run

##### Running AO Inference
This script evaluates the model using the **AO (Answer Only)** format, where the model directly outputs the final answer.

```bash
python inference_AO.py --api_key YOUR_API_KEY --model meta-llama/Llama-3-70b-chat-hf --input_file question_dataset/Dataset/physics_qa.json --output_file question_dataset/Infrence/Llama3_70B/Physics_QA_Llama3_70B_AO.json
```

##### Running COT Inference
This script evaluates the model using the **COT (Chain of Thought)** format, where the model provides a step-by-step solution.

```bash
python inference_COT.py --api_key YOUR_API_KEY --model meta-llama/Llama-3-70b-chat-hf --input_file question_dataset/Dataset/physics_qa.json --output_file question_dataset/Infrence/Llama3_70B/Physics_QA_Llama3_70B_COT.json
```

##### Running Few-Shot Inference
This script evaluates the model using the **(Few-shot Infrence)** learning approach, where a few examples from the same chapter are provided to the model to aid in answering the question.

```bash
python inference_fewshot.py --api_key YOUR_API_KEY --model meta-llama/Llama-3-70b-chat-hf --input_file question_dataset/Dataset/physics_qa.json --output_file question_dataset/Infrence/Llama3_70B/Physics_QA_Llama3_70B_fewshot.json
```

##### Argument Descriptions

Each script accepts the following command-line arguments:

| Argument        | Description                                                   | Example                                      |
|-----------------|---------------------------------------------------------------|----------------------------------------------|
| `--api_key`     | Your Together API key (required)                               | `--api_key "your_api_key_here"`              |
| `--model`       | The model name to use for inference (default provided)         | `--model meta-llama/Llama-3-70b-chat-hf`     |
| `--input_file`  | Path to the input JSON dataset containing the physics QA data  | `--input_file question_dataset/Dataset/physics_qa.json` |
| `--output_file` | Path to save the output results after running the inference    | `--output_file question_dataset/Infrence/Llama3_70B/Physics_QA_Llama3_70B_AO.json` |


### Evaluation
Two Python scripts designed to evaluate model responses to physics questions and extract answers and then Evalute them.

#### Overview

- **Script 1**: `evaluate_responses.py`
  - This script compares model answers to standard answers and determines consistency.
  
- **Script 2**: `extract_answers.py`
  - This script extracts answers from model responses for physics problems.

#### Usage

##### Script 1: `evaluate_responses.py`

This script evaluates the consistency of model responses with standard answers.

##### Command-Line Arguments

- `--api_key`: Your OpenAI API key (required).
- `--model`: The OpenAI model to use for inference (default: `gpt-4o`).
- `--input_file`: Path to the input JSON file containing questions and responses (required).
- `--output_file`: Path to save the final results JSON file (required).

##### Command

```bash
python evaluate_responses.py --api_key "your_openai_api_key" --input_file "path/to/input.json" --output_file "path/to/output.json"
```


##### Script 2: `extract_answers.py`

This script extracts answers from model responses to physics problems.

##### Command-Line Arguments

- `--api_key`: Your OpenAI API key (required).
- `--model`: The OpenAI model to use for extraction (default: `gpt-4`).
- `--input_file`: Path to the input JSON file containing model responses (required).
- `--output_file`: Path to save the extracted answers JSON file (required).

##### Example Command

```bash
python extract_answers.py --api_key "your_openai_api_key" --input_file "path/to/input.json" --output_file "path/to/output.json"
```


