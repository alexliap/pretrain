import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from pretrain.data import download_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# huggingface_hub logs every HTTP request through httpx at INFO
logging.getLogger("httpx").setLevel(logging.ERROR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("data.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/"))
    args = parser.parse_args()

    sources = json.loads(args.config.read_text())

    for name, source in sources.items():
        for crawl in source["crawls"]:
            logger.info("[%s] %s", name, crawl)
            download_dataset(
                source["repo_id"],
                local_dir=args.out_dir / name,
                repo_type=source.get("repo_type", "dataset"),
                allow_patterns=[f"{crawl}/*.parquet"],
            )
