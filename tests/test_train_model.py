from train_model import generate_dataset, split_train_test
import pandas as pd
from scipy.sparse import csr_matrix

def test_generate_dataset_returns_two_series():
    X, y = generate_dataset('data/emotions-train.csv', 'text', 'label')
    assert isinstance(X, pd.Series)
    assert isinstance(y, pd.Series)

def test_split_train_test_returns_expected_output_type():
    X = ["hello I am sad today", "hello I am happy today",
         "hello I am in love today", "hello I am angry today",
         "hello I am scared today", "hello I am surprised today",
         "hello I am delighted today"]
    y = [0, 1, 2, 3, 4, 5, 1]
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    assert isinstance(X_train, csr_matrix)
    assert isinstance(X_test, csr_matrix)
    assert isinstance(y_train, list)
    assert isinstance(y_test, list)

