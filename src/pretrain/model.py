from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def collate_fn(batch, max_seq_length: int):
    """Pad input_ids to the same length within a batch."""
    input_ids = [item["input_ids"][:max_seq_length] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {"input_ids": padded_input_ids}


class MyData:
    def __init__(self, batch_size=2, num_workers=4, max_seq_length=2048, use_packed_data=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.use_packed_data = use_packed_data

    def train_dataloader(self):
        # Load packed or unpacked data based on config
        if self.use_packed_data:
            data_path = f"tokenized_data/packed_train_data_{self.max_seq_length}"
        else:
            data_path = "tokenized_data/train_data"

        dataset = load_from_disk(data_path)
        train = dataset["train"]

        # Remove attention_mask if it exists (may not exist in packed data)
        if "attention_mask" in train.column_names:
            train = train.remove_columns(["attention_mask"])

        train.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            # collate_fn=lambda batch: collate_fn(batch, self.max_seq_length),
        )

    def val_dataloader(self, size: int = int(5e2)):
        # Load packed or unpacked data based on config
        if self.use_packed_data:
            data_path = f"tokenized_data/packed_train_data_{self.max_seq_length}"
        else:
            data_path = "tokenized_data/train_data"

        dataset = load_from_disk(data_path)
        val = dataset["test"]

        # Remove attention_mask if it exists (may not exist in packed data)
        if "attention_mask" in val.column_names:
            val = val.remove_columns(["attention_mask"])

        val = val.select(range(size))
        val.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            # collate_fn=lambda batch: collate_fn(batch, self.max_seq_length),
        )
