import lightning as L
import torch

# import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """Pad input_ids to the same length within a batch."""
    input_ids = [item["input_ids"][:512] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {"input_ids": padded_input_ids}


class MyData(L.LightningDataModule):
    def __init__(self, batch_size=2, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = load_from_disk("tokenized_data/train_data")

        train = (
            dataset["train"].remove_columns(["attention_mask"])
        )

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

        val = (
            dataset["test"].remove_columns(["attention_mask"]).select(range(size))
        )

        val.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )


class MyModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self._load_model()

        self.lr = lr

    @staticmethod
    def _load_model():
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        config.hidden_size = 128
        config.intermediate_size = 1024

        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

        return torch.compile(model)

    def forward(self, idxs):
        return self.model(idxs)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        # For causal LM: input is all tokens except last, target is all tokens except first
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        logits = self.forward(x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        self.log("train_ce_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        logits = self.forward(x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        self.log("val_ce_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer

    # def generate(
    #     self,
    #     query: str,
    #     tokenizer,
    #     max_tokens: int,
    #     temperature: float = 1.0,
    # ):
    #     for _ in range(max_tokens):
    #         tokens = torch.tensor(tokenizer(query)["input_ids"]).reshape(1, -1)
    #         if len(tokens) > 256:
    #             tokens = tokens[-256:]
    #         out = self.forward(tokens) / temperature
    #         out = out[-1, :]
    #         s_out = F.softmax(out, dim=-1)

    #         # do top-k sampling of 50 (huggingface pipeline default)
    #         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    #         topk_probs, _ = torch.topk(s_out, 50, dim=-1)

    #         chosen_token = torch.multinomial(topk_probs, 1)  # (B, 1)

    #         query += tokenizer.decode(chosen_token.item())

    #     return query
