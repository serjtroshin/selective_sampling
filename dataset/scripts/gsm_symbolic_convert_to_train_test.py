from datasets import load_dataset, DatasetDict


def train_test_split(dataset, split_ratio=0.4):
    # print(dataset.train_test_split(test_size=split_ratio, seed=42))
    # split dataset by original_id feature
    test_ids = dataset.unique("original_id")
    test_size = int(len(test_ids) * split_ratio)
    train_ids = test_ids[test_size:]
    test_ids = test_ids[:test_size]
    train = dataset.filter(lambda x: x["original_id"] in train_ids)
    test = dataset.filter(lambda x: x["original_id"] in test_ids)
    # assert original_ids do not intersect between splits
    assert set(train.unique("original_id")) & set(test.unique("original_id")) == set()
    print("original id in train:", train.unique("original_id"))
    print("original id in test:", test.unique("original_id"))
    # split train in train and validation
    # valid_size = 0.1
    # valid_ids = train.unique("original_id")[
    #     : int(len(train.unique("original_id")) * valid_size)
    # ]
    # valid = train.filter(lambda x: x["original_id"] in valid_ids)
    # train = train.filter(lambda x: x["original_id"] not in valid_ids)
    # print("original id in valid:", valid.unique("original_id"))
    # print("original id in train:", train.unique("original_id"))
    return {"train": train, "test": test}


# Load datasets using the correct configuration name
ds_main = load_dataset("apple/GSM-Symbolic", name="main")
# ds_p1 = load_dataset("apple/GSM-Symbolic", name="p1")
# ds_p2 = load_dataset("apple/GSM-Symbolic", name="p2")


# Split datasets
split_main = train_test_split(ds_main["test"])
# split_p1 = train_test_split(ds_p1["test"])
# split_p2 = train_test_split(ds_p2["test"])

# Save each dataset independently
dataset_main = DatasetDict(
    {
        "train": split_main["train"],
        "test": split_main["test"],
        # "valid": split_main["valid"],
    }
)
# dataset_p1 = DatasetDict({"train": split_p1["train"], "test": split_p1["test"]})
# dataset_p2 = DatasetDict({"train": split_p2["train"], "test": split_p2["test"]})


dataset_main.save_to_disk("./gsm_symbolic_main")
# dataset_p1.save_to_disk("./gsm_symbolic_p1")
# dataset_p2.save_to_disk("./gsm_symbolic_p2")

print("Datasets saved successfully in separate folders!")
