import os
from ingest import load_data, save_data
from train_model import generate_dataset, split_train_test, classify
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import logging

logging.basicConfig(filename="logs/main_log.log",
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


dataset_url = "dair-ai/emotion"
output_dir = "data"
file_name = "emotions"
file_path = "data/emotions-train.csv"

if os.path.isfile(file_path):
    print("Dataset already exists. Skipping download")
    pass
else:
    ds, labels = load_data(dataset_url, file_path)
    save_data(ds, output_dir, file_name)

X, y = generate_dataset(file_path, 'text', 'label')
X_train, X_test, y_train, y_test = split_train_test(X, y)
y_pred = classify(X_train, X_test, y_train)

result_confusion_matrix = confusion_matrix(y_test, y_pred)
logger.info(f'Confusion matrix: {result_confusion_matrix}')

accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

logger.info(f'''Accuracy: {accuracy}; F1-score: {f1_score}; Recall: {recall}''')