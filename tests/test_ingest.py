from datasets import Dataset, DatasetDict
from ingest import load_data, save_data
import pandas as pd

dataset_url = "dair-ai/emotion"

def test_load_data_returns_dataset():
    ds, label = load_data(dataset_url)
    assert isinstance(ds, DatasetDict)
    assert isinstance(label, list)

def test_save_data_creates_csv_file(tmp_path): 

    ds = DatasetDict({
        "train": Dataset.from_dict({
            "text": ["happy", "sad"],
            "label": [1, 0],
        })
    })

    save_data(ds, tmp_path, 'emotions')

    csv_file = tmp_path / "emotions-train.csv"
    print(list(tmp_path.iterdir()))
    assert csv_file.exists()

    df = pd.read_csv(csv_file)
    assert list(df.columns) == ["text", "label"]
