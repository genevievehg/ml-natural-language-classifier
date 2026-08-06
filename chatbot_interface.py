import logging
from chatbot import Chatbot
from classifier import EmotionClassifier

logging.basicConfig(filename="logs/chatbot_log.log",
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def chatbot_interface():
    print('''Chatbot Assistant: Hey! I'm here to help! \nWhat text would you like me to classify?
You can type 'exit' to leave the chat. \nUser:''')
    prompt = input()
    while prompt != 'exit':
        emotion_classifier = EmotionClassifier('data/models/trained_classifier.pkl')
        emotion_classification = emotion_classifier.classify_text(prompt)
        logger.info(f'''User prompt: {prompt}''')
        logger.info(f'''Result of emotion classification: {emotion_classification}''')
        appended_prompt = f'''\n<|user|>\n {prompt}.
        Emotion classification: {emotion_classification} \n<|end|>\n<|assistant|>\n'''
        logger.info(f'''Second prompt: {appended_prompt}''')
        chatbot = Chatbot('''<|system|>\nYou are a friendly assistant.
        You will be given a user's message and an emotion classification result.
        Answer the user's question taking the classification result into account.<|end|>\n''')
        reply = chatbot.generate_reply(appended_prompt)
        logger.info(f'''Chatbot reply: {reply}''')
        print(f"Bot:\n{reply}\n")
        print("User:")
        prompt = input()

if __name__ == "__main__":
    chatbot_interface()