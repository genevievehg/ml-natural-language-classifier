import pickle

class EmotionClassifier:
    def __init__(self, classifier_path):
        with open(classifier_path, 'rb') as f:
            self.model = pickle.load(f)

    def classify_text(self, text):
        with open('data/models/fitted_vectoriser.pkl', 'rb') as f:
            self.vectoriser = pickle.load(f)
        text_vec = self.vectoriser.transform([text])
        classification = self.model.predict(text_vec)
        emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        classification_int = classification[0]
        emotion = emotions[classification_int]
        return emotion