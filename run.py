
import argparse
import importlib
import sys
from pathlib import Path

# чтобы работало и вне PyCharm
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nlpkit.config import load_config, set_seed  # noqa: E402

STEPS = [f"pr{i}" for i in range(1, 8)]
EXTRA = ["download-wiki", "download-kk", "download-arxiv", "make-eval", "eval"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=STEPS + ["all"] + EXTRA)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    if args.step == "download-wiki":
        from nlpkit.wiki import download
        download(cfg)
        return
    if args.step == "download-kk":
        from nlpkit.wiki import download_kk
        download_kk(cfg)
        return
    if args.step == "download-arxiv":
        from nlpkit.arxiv_fetch import download
        download(cfg)
        return
    if args.step == "make-eval":
        from nlpkit.regex_eval import make_eval_set
        make_eval_set(cfg)
        return
    if args.step == "eval":
        from nlpkit.regex_eval import evaluate
        evaluate(cfg)
        return

    for step in (STEPS if args.step == "all" else [args.step]):
        module = importlib.import_module(f"experiments.{step}")
        module.main(cfg)


if __name__ == "__main__":
    main()