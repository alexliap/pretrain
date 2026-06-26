import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def collate_fn(batch, max_seq_length: int, return_attention_mask: bool = False):
    """Pad input_ids to the same length within a batch.

    When return_attention_mask is True (e.g. for non-packed validation data),
    an attention_mask is built from the real sequence lengths so the model does
    not attend to padding.
    """
    input_ids = [item["input_ids"][:max_seq_length] for item in batch]
    lengths = torch.tensor([len(x) for x in input_ids])
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    out = {"input_ids": padded_input_ids}
    if return_attention_mask:
        positions = torch.arange(padded_input_ids.size(1))
        out["attention_mask"] = (positions[None, :] < lengths[:, None]).long()
    return out


class PretrainDataLoader:
    def __init__(
        self, batch_size=2, num_workers=4, max_seq_length=2048, use_packed_data=True
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.use_packed_data = use_packed_data

    def train_dataloader(self):
        # Load packed or unpacked data based on config
        if self.use_packed_data:
            data_path = f"tokenized_data/packed_train_data_{self.max_seq_length}"
        else:
            data_path = "tokenized_data/train"

        dataset = load_from_disk(data_path)
        train = dataset["train"]

        # Remove attention_mask if it exists (may not exist in packed data)
        if "attention_mask" in train.column_names:
            train = train.remove_columns(["attention_mask"])

        train.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(
                batch, self.max_seq_length, return_attention_mask=True
            ),
        )

    def val_dataloader(self, size: int = int(5e2)):
        # Load packed or unpacked data based on config
        # if self.use_packed_data:
        #     data_path = f"tokenized_data/packed_train_data_{self.max_seq_length}"
        # else:
        data_path = "tokenized_data/test"

        val = load_from_disk(data_path)
        # val = dataset["test"]

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
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(
                batch, self.max_seq_length, return_attention_mask=True
            ),
        )
