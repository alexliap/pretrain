from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def collate_fn(batch):
    """Pad input_ids to the same length within a batch."""
    input_ids = [item["input_ids"][:512] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {"input_ids": padded_input_ids}


class MyData:
    def __init__(self, batch_size=2, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = load_from_disk("tokenized_data/train_data")

        train = dataset["train"].remove_columns(["attention_mask"])

        train.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self, size: int = int(5e2)):
        dataset = load_from_disk("tokenized_data/train_data")

        val = dataset["test"].remove_columns(["attention_mask"]).select(range(size))

        val.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
