"""Continued pre-training (CPT) of LFM3-1.7B on the Greek Common Crawl corpus.

Follows Liquid's lfm3_cpt_with_trl.ipynb: same model, LoRA and SFTConfig settings,
on plain text packed into full rows. Greek replaces Turkish, so:

* the corpus is every crawl of alexliap/greek-cc rather than Turkish Wikipedia;
* the out-of-distribution set flips to Greek Wikipedia, since we now train on web
  pages;
* the Greek benchmarks in pretrain.evaluation replace the notebook's TurBLiMP
  grammar test, which is Turkish-only, and also run during training.

Launch from the repo root with train.sh, or:
    uv run accelerate launch scripts/lfm3_greek/cpt.py data.max_train_docs=3000
"""

import math
from pathlib import Path

import hydra
import torch
import trackio
from accelerate import PartialState
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from pretrain.config import EvaluationConfig, EvaluationTaskConfig, LoraConfig
from pretrain.evaluation import EvaluationResult, EvaluationRunner
from pretrain.training.post_training_common import TrackioCallback, build_peft_config

load_dotenv()
SEED = 0
set_seed(SEED)


def stream_docs(dataset_id: str, n_docs: int, config_name: str | None = None) -> list:
    """The first `n_docs` documents of a shuffled stream, downloading only those."""
    stream = load_dataset(dataset_id, name=config_name, split="train", streaming=True)
    stream = stream.select_columns(["text"]).shuffle(seed=42, buffer_size=10_000)
    return list(stream.take(n_docs))


def load_corpus(data: dict) -> tuple[Dataset, Dataset]:
    """The (train, held-out) split of greek-cc.

    Crawls are deduplicated separately, not against each other, so a held-out page
    can have near-duplicates in training and its perplexity may look optimistic.
    The Wikipedia set is the more honest curve.
    """
    n_eval, max_train = data["num_eval_docs"], data["max_train_docs"]
    if max_train is not None:
        docs = stream_docs(data["dataset_id"], n_eval + max_train)
        return Dataset.from_list(docs[n_eval:]), Dataset.from_list(docs[:n_eval])

    corpus = (
        load_dataset(data["dataset_id"], split="train")
        .shuffle(SEED)
        .select_columns(["text"])
    )
    # Contiguous ranges, since a shuffled split would add an indices mapping that
    # TRL's tokenization then has to read through at random.
    n_train = len(corpus) - n_eval
    return corpus.select(range(n_train)), corpus.select(range(n_train, len(corpus)))


def perplexities(metrics: dict) -> dict[str, float]:
    """Perplexity per eval set, from Trainer's eval_<name>_loss metrics."""
    return {
        f"{key.removesuffix('_loss')}_perplexity": math.exp(value)
        for key, value in metrics.items()
        if key.endswith("_loss")
    }


class PerplexityCallback(TrainerCallback):
    """Log each eval set's perplexity next to the loss TrackioCallback logs."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and state.is_world_process_zero:
            trackio.log(perplexities(metrics), step=state.global_step)


class GreekBenchmarkCallback(TrainerCallback):
    """Run the Greek benchmarks every `every` steps.

    Hooks on_step_end rather than on_evaluate, which fires once per eval set.
    """

    def __init__(self, run_benchmarks, every: int):
        self.run_benchmarks = run_benchmarks
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            self.run_benchmarks(state.global_step)


def print_summary(
    before: dict, after: dict, bench_before: list, bench_after: list
) -> None:
    """Before/after table, as at the end of the notebook. A task that errored out
    (see EvaluationTask.run) or a resumed run's missing baseline shows as "-"."""

    def flatten(metrics: dict, results: list[EvaluationResult]) -> dict[str, float]:
        flat = perplexities(metrics)
        for result in results:
            for name, value in result.metrics.items():
                flat[f"{result.task_name}/{name}"] = value
        return flat

    original, finetuned = flatten(before, bench_before), flatten(after, bench_after)

    def cell(scores: dict, name: str) -> str:
        return f"{scores[name]:.4f}" if name in scores else "-"

    print(f"{'':<36}{'original':>10}{'fine-tuned':>12}")
    for name in sorted(original.keys() | finetuned.keys()):
        print(f"{name:<36}{cell(original, name):>10}{cell(finetuned, name):>12}")


@hydra.main(version_base=None, config_path="configs", config_name="cpt")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)

    # LFM3-1.7B is released in bf16. GPUs without bf16 support load it in fp16
    # instead, and the LoRA adapters train in float32 either way. No GPU at all
    # only happens in CPU smoke tests.
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported(including_emulation=False)
    use_fp16 = has_cuda and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    args = SFTConfig(
        **config["training"],
        bf16=use_bf16,
        fp16=use_fp16,
        run_name=config["run_name"],
        # trackio is driven by TrackioCallback, see its docstring.
        report_to="none",
    )

    # A token budget is a step count in disguise: packed rows all have max_length
    # tokens. Going through max_steps keeps warmup and the cosine schedule spanning
    # the whole budget, which stopping early from a callback would cut short.
    if config["total_tokens"] is not None:
        tokens_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.world_size
            * args.max_length
        )
        args.max_steps = max(1, config["total_tokens"] // tokens_per_step)

    # Only the main process talks to trackio: one shared run, not one per rank.
    state = PartialState()
    if state.is_main_process:
        trackio.init(
            project=config["logging"]["project_name"],
            auto_log_gpu=config["logging"]["auto_log_gpu"],
            name=config["run_name"],
            config=config,
            space_id=None,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=True
    )
    # No device_map="auto" as in the notebook: it splits the model across GPUs,
    # which breaks DDP under accelerate launch. Trainer places the model instead.
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], dtype=dtype, trust_remote_code=True
    )

    data = config["data"]
    with state.main_process_first():
        train_dataset, web_dataset = load_corpus(data)
        wikipedia_dataset = Dataset.from_list(
            stream_docs(
                data["ood_dataset_id"], data["num_ood_docs"], data["ood_config"]
            )
        )

    evaluation = dict(config["evaluation"])
    tasks = {
        name: EvaluationTaskConfig(**task)
        for name, task in evaluation.pop("tasks").items()
    }
    evaluation_config = EvaluationConfig(tasks=tasks, **evaluation)

    def run_benchmarks(step: int) -> list[EvaluationResult]:
        # Rank 0 only: tasks log straight to trackio, which only rank 0 has
        # initialized. The other ranks wait at the next gradient sync. Autocast
        # because this runs outside the Trainer's forward, where the bf16 base
        # and float32 adapters would otherwise mix.
        if not trainer.is_world_process_zero():
            return []
        with trainer.accelerator.autocast():
            return runner.run_all(trainer.model, step)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset={"web": web_dataset, "wikipedia": wikipedia_dataset},
        processing_class=tokenizer,
        peft_config=build_peft_config(LoraConfig(**config["lora"])),
        callbacks=[
            TrackioCallback(),
            PerplexityCallback(),
            GreekBenchmarkCallback(run_benchmarks, config["benchmark_steps"]),
        ],
    )
    trainer.model.print_trainable_parameters()

    runner = EvaluationRunner(evaluation_config, tokenizer, trainer.accelerator)

    resume_from = config["resume_from_checkpoint"]
    baseline, baseline_benchmarks = {}, []
    if resume_from is None:
        baseline = trainer.evaluate()
        baseline_benchmarks = run_benchmarks(0)

    trainer.train(resume_from_checkpoint=resume_from)

    final = trainer.evaluate()
    final_benchmarks = run_benchmarks(trainer.state.global_step)
    if trainer.is_world_process_zero():
        print_summary(baseline, final, baseline_benchmarks, final_benchmarks)

    output_dir = Path(args.output_dir)
    trainer.save_model(str(output_dir / "lora"))
    if trainer.is_world_process_zero():
        # save_pretrained also copies LFM3's modeling code next to the weights, so
        # the merged model loads with trust_remote_code=True like the original.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output_dir / "merged")
        tokenizer.save_pretrained(output_dir / "merged")
        trackio.finish()


if __name__ == "__main__":
    main()
