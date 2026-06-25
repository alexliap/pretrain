"""Command line interface for the pretrain package."""

import argparse
import logging

from pretrain.data import download_dataset, tokenize_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
        help="Download a dataset from the Hugging Face Hub into a local directory.",
    )
    parser.add_argument(
        "repo_id",
        help="Hub repository id, e.g. 'alexliap/high-quality-gr-text'.",
    )
    parser.add_argument(
        "local_dir",
        help="Directory the repository files are downloaded into.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Type of the Hub repository (default: dataset).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision (branch, tag or commit sha) to download.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Glob patterns; only matching files are downloaded.",
    )
    parser.set_defaults(func=_run_download)


def _add_tokenize_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "tokenize",
        help="Tokenize a parquet dataset and save the train/test splits to disk.",
    )
    parser.add_argument(
        "--tokenizer-repo-id",
        required=True,
        help="Hub repository id or local path of the tokenizer.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the parquet file(s) to tokenize.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of the dataset held out for the test split (default: 0.1).",
    )
    parser.add_argument(
        "--output-path",
        default="tokenized_data/",
        help="Directory the tokenized dataset is saved into.",
    )
    parser.set_defaults(func=_run_tokenize)


def _run_download(args: argparse.Namespace) -> None:
    download_dataset(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
    )

def _run_tokenize(args: argparse.Namespace) -> None:
    tokenize_dataset(
        tokenizer_repo_id=args.tokenizer_repo_id,
        data_path=args.data_path,
        test_size=args.test_size,
        output_path=args.output_path,
    )


def main() -> None:
    """Entry point for the ``pretrain-data`` command."""
    parser = argparse.ArgumentParser(
        prog="pretrain-data",
        description="Data preparation commands for the pretrain package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_download_parser(subparsers)
    _add_tokenize_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
