import logging
from chatbot import Chatbot
from classifier import EmotionClassifier
from sentence_transformers import SentenceTransformer
from retriever import Retriever

logging.basicConfig(filename="logs/chatbot_log.log",
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def chatbot_interface():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    retriever = Retriever('data/facts.txt')
    emotion_classifier = EmotionClassifier('data/models/trained_classifier.pkl')
    chatbot = Chatbot()

    print('''Chatbot Assistant: Hey! I'm here to help! \nWhat text would you like me to classify?
You can type 'exit' to leave the chat. \nUser:''')
    prompt = input()
    while prompt != 'exit':
        emotion_classification = emotion_classifier.classify_text(prompt)

        embeddings = model.encode(prompt)

        retrieved_chunks = retriever.retrieve_top_3_chunks(embeddings)

        logger.info(f'''User prompt: {prompt}''')
        logger.info(f'''Result of emotion classification: {emotion_classification}''')
        logger.info(f'''Retrieved chunks: {retrieved_chunks}''')
        
        reply = chatbot.generate_reply(prompt, emotion_classification, retrieved_chunks)

        logger.info(f'''Chatbot reply: {reply}''')

        print(f"Bot:\n{reply}\n")
        print(f'''Do you have any other text to classify?" \nRemember you can type 'exit' to leave the chat. \nUser:''')
        prompt = input()

if __name__ == "__main__":
    chatbot_interface()