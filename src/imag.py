import argparse
from pathlib import Path

import torch

from train import cross_validate, rank_candidates


ROOT = Path(__file__).resolve().parents[1]


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gene-feature",
        type=Path,
        default=ROOT / "data/features/gene_signed_sig.csv",
    )
    parser.add_argument(
        "--isoform-feature",
        type=Path,
        default=ROOT / "data/features/isoform_signed_sig.csv",
    )
    parser.add_argument(
        "--ratio-feature",
        type=Path,
        default=ROOT / "data/features/ratio_signed_sig.csv",
    )
    parser.add_argument(
        "--gene-graph",
        type=Path,
        default=ROOT / "data/graphs/gene_graph.csv",
    )
    parser.add_argument(
        "--isoform-graph",
        type=Path,
        default=ROOT / "data/graphs/isoform_graph.csv",
    )
    parser.add_argument(
        "--ratio-graph",
        type=Path,
        default=ROOT / "data/graphs/ratio_graph.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/labels/ad_gene_labels.csv",
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iMAG-AD")
    commands = parser.add_subparsers(dest="command", required=True)

    cv_parser = commands.add_parser("cv", help="run cross-validation")
    add_data_arguments(cv_parser)
    add_training_arguments(cv_parser)
    cv_parser.add_argument("--n-splits", type=int, default=5)
    cv_parser.add_argument(
        "--fold", type=int, default=None, help="run only one fold"
    )
    cv_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/cross_validation",
    )
    cv_parser.set_defaults(handler=cross_validate)

    rank_parser = commands.add_parser("rank", help="rank candidate genes")
    add_data_arguments(rank_parser)
    add_training_arguments(rank_parser)
    rank_parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/ranking"
    )
    rank_parser.set_defaults(handler=rank_candidates)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
