from datasets import load_dataset

def load_data(url):
    ds = load_dataset(url, "unsplit")
    labels = ds["train"].features["label"].names
    return ds, labels

def save_data(ds, output_dir, file_name):
    for split, dataset in ds.items():
        dataset.to_csv(f"{output_dir}/{file_name}-{split}.csv", index=None)