from datasets import load_dataset, DatasetDict


def train_test_split(dataset, split_ratio=0.4):
    train_test = dataset.train_test_split(test_size=split_ratio, seed=42)
    train = train_test["train"]
    test = train_test["test"]

    # print number of samples per category
    unique_categories_train = train.unique("category")
    unique_categories_test = test.unique("category")
    count_train = {
        cat: train.filter(lambda x: x["category"] == cat).num_rows
        for cat in unique_categories_train
    }
    count_test = {
        cat: test.filter(lambda x: x["category"] == cat).num_rows
        for cat in unique_categories_test
    }
    print("Number of samples per category in train:", count_train)
    print("Number of samples per category in test:", count_test)

    return {"train": train, "test": test}


# Load datasets using the correct configuration name
ds_main = load_dataset("TIGER-Lab/MMLU-Pro")


# Split datasets
split_main = train_test_split(ds_main["test"])
split_validation = ds_main["validation"]
# split_p1 = train_test_split(ds_p1["test"])
# split_p2 = train_test_split(ds_p2["test"])

# Save each dataset independently
dataset_main = DatasetDict(
    {
        "validation": split_validation,
        "train": split_main["train"],
        "test": split_main["test"],
        # "valid": split_main["valid"],
    }
)

dataset_main.save_to_disk("./mmlu_pro")

print("Datasets saved successfully in separate folders!")
