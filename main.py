"""Executa a análise completa ou apenas uma questão.

Uso:
    python main.py              # todas as questões
    python main.py --questao 4  # apenas a questão 4
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janelas

from src import auto_regression, boston_regression, prediction_intervals  # noqa: E402
from src.config import FIG_DIR, TABLE_DIR, ensure_dirs  # noqa: E402

QUESTOES = {
    2: auto_regression.run,
    3: boston_regression.run,
    4: prediction_intervals.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questao", type=int, choices=QUESTOES, help="questão a executar")
    args = parser.parse_args()

    ensure_dirs()
    for q in [args.questao] if args.questao else QUESTOES:
        QUESTOES[q]()

    print(f"\nTabelas salvas em: {TABLE_DIR}")
    print(f"Figuras salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()
