"""Questão 2: regressão linear múltipla na base Auto (resposta: mpg)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .data import load_auto
from .utils import header, save_figure, save_table

PREDITORES = ["cylinders", "displacement", "horsepower", "weight",
              "acceleration", "year", "origin"]
FORMULA_ORIGINAL = "mpg ~ " + " + ".join(PREDITORES)
FORMULA_LOG = ("mpg ~ cylinders + displacement + np.log(horsepower) + np.log(weight)"
               " + acceleration + year + origin")
FORMULA_QUAD = ("mpg ~ cylinders + displacement + horsepower + I(horsepower**2)"
                " + weight + I(weight**2) + acceleration + year + origin")


def plot_scatter_matrix(df: pd.DataFrame) -> None:
    axes = pd.plotting.scatter_matrix(df, figsize=(16, 16), diagonal="hist",
                                      s=8, alpha=0.5)
    fig = axes[0, 0].get_figure()
    fig.suptitle("(a) Matriz de dispersão - Auto", y=0.92, fontsize=16)
    save_figure(fig, "q2a_auto_scatter_matrix", dpi=80)


def plot_correlation(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("(b) Matriz de correlação - Auto")
    fig.tight_layout()
    save_figure(fig, "q2b_auto_correlacao")


def run() -> dict:
    header("QUESTÃO 2 - Regressão linear múltipla (Auto, resposta: mpg)")
    auto = load_auto()
    auto_num = auto.drop(columns=["name"])
    print(f"Observações: {len(auto)}")

    # (a) Dispersão
    plot_scatter_matrix(auto_num)

    # (b) Correlação
    corr = auto_num.corr()
    save_table(corr.round(3), "q2b_auto_correlacao")
    plot_correlation(corr)
    print("\n(b) Correlação de cada variável com mpg:")
    print(corr["mpg"].drop("mpg").sort_values().round(3).to_string())

    # (c) Regressão múltipla
    modelo = smf.ols(FORMULA_ORIGINAL, data=auto).fit()
    print("\n(c) Modelo completo")
    print(modelo.summary())

    coefs = pd.DataFrame({
        "coef": modelo.params, "erro_padrao": modelo.bse,
        "t": modelo.tvalues, "pvalor": modelo.pvalues,
    })
    coefs["significativo_1%"] = coefs["pvalor"] < 0.01
    save_table(coefs.round(6), "q2c_auto_coeficientes")

    sig = coefs.drop("Intercept").query("pvalor < 0.01").index.tolist()
    print(f"\nTeste F global: F = {modelo.fvalue:.1f}, p-valor = {modelo.f_pvalue:.2e}")
    print(f"Preditores significativos a 1%: {', '.join(sig)}")
    print(f"Coeficiente de year: {modelo.params['year']:.4f} mpg por ano")

    # (d) Transformações
    modelos = {
        "Original": modelo,
        "Com log(horsepower), log(weight)": smf.ols(FORMULA_LOG, data=auto).fit(),
        "Com horsepower², weight²": smf.ols(FORMULA_QUAD, data=auto).fit(),
    }
    comparacao = pd.DataFrame({
        "R2": {k: m.rsquared for k, m in modelos.items()},
        "R2_ajustado": {k: m.rsquared_adj for k, m in modelos.items()},
        "AIC": {k: m.aic for k, m in modelos.items()},
    }).round(4)
    save_table(comparacao, "q2d_auto_transformacoes")
    print("\n(d) Comparação de transformações:")
    print(comparacao.to_string())

    quad = modelos["Com horsepower², weight²"]
    print("\nP-valores dos termos quadráticos:")
    print(quad.pvalues.filter(like="** 2").round(6).to_string())

    return {"modelo": modelo, "comparacao": comparacao}


if __name__ == "__main__":
    from .config import ensure_dirs
    ensure_dirs()
    run()
