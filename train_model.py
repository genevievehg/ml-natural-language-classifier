import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def generate_dataset(data_path, column_X, column_y):
    df = pd.read_csv(data_path)
    X = df[column_X]
    y = df[column_y]
    return X, y

def split_train_test(X, y):
    vectoriser = TfidfVectorizer(max_features=5000)
    X_vec = vectoriser.fit_transform(X)
    with open('data/models/fitted_vectoriser.pkl', 'wb') as f:
            pickle.dump(vectoriser, f)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.7, random_state=0)
    return X_train, X_test, y_train, y_test

def classify(X_train, X_test, y_train):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    with open('data/models/trained_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    return y_pred

