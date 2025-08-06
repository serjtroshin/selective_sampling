from datasets import load_dataset, DatasetDict


def train_test_split(dataset, split_ratio=0.5):
    return dataset.train_test_split(test_size=split_ratio, seed=42)


# Load datasets using the correct configuration name
ds_main = load_dataset("google/IFEval")


# Split datasets
split_main = train_test_split(ds_main["train"])

# Save each dataset independently
dataset_main = DatasetDict({"train": split_main["train"], "test": split_main["test"]})
dataset_main.save_to_disk("./ifeval")

print("Datasets saved successfully in separate folders!")
