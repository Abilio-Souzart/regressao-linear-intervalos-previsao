"""Questão 4: intervalos de previsão para crim (Boston).

Compara três métodos em um mesmo conjunto de teste:
- intervalo de previsão tradicional da regressão linear (OLS);
- bootstrap percentil (pairs bootstrap + reamostragem de resíduos);
- split conformal prediction.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from .config import ALPHA, N_BOOTSTRAP, SEED
from .data import load_boston
from .utils import header, save_figure, save_table


# ---------------------------------------------------------------------------
# Métodos
# ---------------------------------------------------------------------------
def design_matrix(df: pd.DataFrame, preditores: list[str]) -> np.ndarray:
    return sm.add_constant(df[preditores], has_constant="add").to_numpy(float)


def intervalo_tradicional(X_train, y_train, X_new, alpha=ALPHA):
    """Intervalo de previsão clássico, assumindo erros normais e homocedásticos."""
    fit = sm.OLS(y_train, X_train).fit()
    frame = fit.get_prediction(X_new).summary_frame(alpha=alpha)
    return (frame["mean"].to_numpy(), frame["obs_ci_lower"].to_numpy(),
            frame["obs_ci_upper"].to_numpy())


def bootstrap_so_coeficientes(X_train, y_train, X_new, B=N_BOOTSTRAP,
                              alpha=ALPHA, seed=SEED):
    """Implementação ORIGINAL do notebook (mantida apenas para comparação).

    Reamostra o treino, reajusta o modelo e tira percentis das previsões
    X_new @ beta*. Isso só captura a incerteza na estimativa dos coeficientes,
    ou seja, é um intervalo de CONFIANÇA para a média E[Y | X], e não um
    intervalo de PREVISÃO para uma nova observação Y.
    """
    rng = np.random.default_rng(seed)
    n = len(y_train)
    preds = np.empty((len(X_new), B))
    for b in range(B):
        idx = rng.integers(0, n, n)
        beta, *_ = np.linalg.lstsq(X_train[idx], y_train[idx], rcond=None)
        preds[:, b] = X_new @ beta
    lower, upper = np.percentile(preds, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
    return preds.mean(axis=1), lower, upper


def intervalo_bootstrap(X_train, y_train, X_new, B=N_BOOTSTRAP,
                        alpha=ALPHA, seed=SEED):
    """Intervalo de previsão por bootstrap percentil (versão corrigida).

    Uma nova observação é Y_new = X_new @ beta + erro. O intervalo precisa
    refletir as duas fontes de incerteza:
      1. incerteza nos coeficientes -> reajustar o modelo em amostras bootstrap;
      2. ruído da nova observação   -> somar um resíduo sorteado a cada previsão.

    Os resíduos de treino subestimam o erro fora da amostra, então são
    inflados por sqrt(n / (n - p)), a mesma correção de graus de liberdade
    usada no estimador não viesado da variância.
    """
    rng = np.random.default_rng(seed)
    n, p = X_train.shape
    m = len(X_new)
    inflacao = np.sqrt(n / (n - p))
    sims = np.empty((m, B))
    for b in range(B):
        idx = rng.integers(0, n, n)
        Xb, yb = X_train[idx], y_train[idx]
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        residuos = (yb - Xb @ beta) * inflacao
        sims[:, b] = X_new @ beta + rng.choice(residuos, size=m, replace=True)
    lower, upper = np.percentile(sims, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)

    beta_full, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_new @ beta_full, lower, upper


def intervalo_conformal(X_train, y_train, X_cal, y_cal, X_new, alpha=ALPHA):
    """Split conformal com escore |y - y_hat| calculado na base de calibração."""
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    escores = np.abs(y_cal - X_cal @ beta)
    n_cal = len(escores)
    nivel = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_hat = np.quantile(escores, nivel, method="higher")
    pred = X_new @ beta
    return pred, pred - q_hat, pred + q_hat, q_hat


def avaliar(y, lower, upper) -> dict:
    return {
        "cobertura": float(np.mean((y >= lower) & (y <= upper))),
        "largura_media": float(np.mean(upper - lower)),
        "largura_mediana": float(np.median(upper - lower)),
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def plot_intervalos(y_test, resultados: dict) -> None:
    ordem = np.argsort(resultados["Tradicional (OLS)"]["pred"])
    n = len(resultados)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True, sharey=True)
    x = np.arange(len(y_test))
    for ax, (nome, r) in zip(axes, resultados.items()):
        lo, up, pr, y = r["lower"][ordem], r["upper"][ordem], r["pred"][ordem], y_test[ordem]
        dentro = (y >= lo) & (y <= up)
        ax.vlines(x, lo, up, color="#9bb7d4", lw=2, label="Intervalo de 95%")
        ax.plot(x, pr, "_", color="#1f4e79", ms=8, label="Previsão")
        ax.plot(x[dentro], y[dentro], "o", color="#2e8b57", ms=4, label="Observado (dentro)")
        ax.plot(x[~dentro], y[~dentro], "o", color="#c0392b", ms=5, label="Observado (fora)")
        m = avaliar(y_test, r["lower"], r["upper"])
        ax.set_title(f"{nome}: cobertura = {m['cobertura']:.1%}, "
                     f"largura média = {m['largura_media']:.2f}")
        ax.set_ylabel("crim")
    axes[0].legend(loc="upper left", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Observações de teste (ordenadas pela previsão)")
    fig.tight_layout()
    save_figure(fig, "q4_intervalos_previsao")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def run() -> pd.DataFrame:
    header("QUESTÃO 4 - Intervalos de previsão para crim (Boston)")
    boston = load_boston()
    preditores = [c for c in boston.columns if c != "crim"]

    # (a) Divisão 70% treino / 15% calibração / 15% teste
    train, temp = train_test_split(boston, test_size=0.30, random_state=SEED)
    cal, test = train_test_split(temp, test_size=0.50, random_state=SEED)
    print(f"Treino: {len(train)} | Calibração: {len(cal)} | Teste: {len(test)}")

    X_tr, y_tr = design_matrix(train, preditores), train["crim"].to_numpy()
    X_cal, y_cal = design_matrix(cal, preditores), cal["crim"].to_numpy()
    X_te, y_te = design_matrix(test, preditores), test["crim"].to_numpy()

    resultados = {}

    # (b) Tradicional
    pred, lo, up = intervalo_tradicional(X_tr, y_tr, X_te)
    resultados["Tradicional (OLS)"] = {"pred": pred, "lower": lo, "upper": up}

    # (c) Bootstrap: versão original (incorreta) e corrigida
    pred, lo, up = bootstrap_so_coeficientes(X_tr, y_tr, X_te)
    original = {"pred": pred, "lower": lo, "upper": up}
    pred, lo, up = intervalo_bootstrap(X_tr, y_tr, X_te)
    resultados["Bootstrap percentil"] = {"pred": pred, "lower": lo, "upper": up}

    # (d) Split conformal
    pred, lo, up, q_hat = intervalo_conformal(X_tr, y_tr, X_cal, y_cal, X_te)
    resultados["Split conformal"] = {"pred": pred, "lower": lo, "upper": up}
    print(f"q_hat (split conformal): {q_hat:.4f}")

    # (e) Comparação
    tabela = pd.DataFrame({nome: avaliar(y_te, r["lower"], r["upper"])
                           for nome, r in resultados.items()}).T
    tabela.loc["Bootstrap original (só coeficientes)"] = avaliar(
        y_te, original["lower"], original["upper"])
    tabela["% limite inferior < 0"] = [
        np.mean(r["lower"] < 0) for r in resultados.values()
    ] + [np.mean(original["lower"] < 0)]
    tabela = tabela.round(4)
    save_table(tabela, "q4_comparacao_intervalos")
    print("\n(e) Comparação dos métodos (nível nominal de 95%):")
    print(tabela.to_string())

    plot_intervalos(y_te, resultados)
    return tabela


if __name__ == "__main__":
    from .config import ensure_dirs
    ensure_dirs()
    run()
