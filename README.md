# Natural Language Emotion Classifier & Chatbot 🗃️🤖

This project builds a Chatbot using text classification, retrieval-augmented generation, and a large language model (LLM).

## Project Overview

User input is processed as follows:
1. Logistic Regression is used to identify the emotion in the user message.
2. Sentence transformers are used to convert text to numerical vector repressentations.
3. Cosine similarity is used to retrieve the most relevant contextual information.
4. The prompt for the LLM is contructed from the user's message, the classification result, and the additional context.
5. The TinyLlama model is used to generate a response.


Additional features are as follows:
- Evaluation of the classification result using a Confusion Matrix, Accuracy, F1-Score, and Recall.
- Separate interfaces for classification and the chatbot.
- Logging of classification results and chatbot interactions.

## Requirements
- Python 3.13+

## Setup
1. Clone the repo
2. Create and activate a venv in the root directory:

    `python -m venv .venv`

    `source .venv/bin/activate`

3. Install project requirements:

    `pip install -r requirements.txt`

4. Run 

    `python main.py`
   
   This will (1) load and save the dataset, (2) split it into train/test sets, (3) save the vectoriser, (4) train and save the classifier, and (5) output evaluation metrics.
  
To use the simple classification interface, run: 

`python simple_interface.py`

Here, the user can enter some text for classification and receive a result of one of six emotions (sad, happiness, angry, love, fear, surprise).

To use the Chatbot, run:

`python chatbot_interface.py`

Here, the user can interact with the LLM and receive more meaningful responses supported by additional context.

## Limitations

There are several areas for improvement:
- Different LLM models could be explored to generate improved responses. These could not be pursued due to hardware restrictions.
- Different chunking strategies could be explored. A concious decision was made to separate chunks by new line here due to the limited size of the facts themselves.

## Dataset

This project uses the **Emotion** dataset from Hugging Face:

- Dataset: https://huggingface.co/datasets/dair-ai/emotion
- Original paper:
  > Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).
  > *CARER: Contextualized Affect Representations for Emotion Recognition.*
  > Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 3687–3697.