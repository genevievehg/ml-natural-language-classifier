from classifier import EmotionClassifier
from sklearn.linear_model import LogisticRegression

emotion_classifier = EmotionClassifier('data/models/trained_classifier.pkl')

def test_classifier_model_is_logistic_regression_model():
    assert isinstance(emotion_classifier.model, LogisticRegression)

def test_classify_text_returns_emotion_from_list():
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    result = emotion_classifier.classify_text('example_text')
    assert result in emotions

def test_classify_text_returns_emotion_from_list_example_2():
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    result = emotion_classifier.classify_text('a & b 4ever')
    assert result in emotions