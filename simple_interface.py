from classifier import EmotionClassifier


def interface():
    print('''**Emotion Classifier**\n
To exit the classifier, please type 'exit'\n
Enter text for emotion classification:''')
    user_input = input()
    while user_input != 'exit':
        emotion_classifier = EmotionClassifier('data/models/trained_classifier.pkl')
        emotion_classification = emotion_classifier.classify_text(user_input)
        print(f'Emotion: {emotion_classification}')
        print('Enter text for emotion classification:')
        user_input = input()

if __name__ == "__main__":
    interface()