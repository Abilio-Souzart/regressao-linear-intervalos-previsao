"""Questão 3: regressões para a taxa de criminalidade (crim) na base Boston."""
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from .data import load_boston
from .utils import header, save_figure, save_table

NIVEL = 0.05


def regressoes_simples(df: pd.DataFrame, preditores: list[str]) -> pd.DataFrame:
    linhas = []
    for var in preditores:
        m = smf.ols(f"crim ~ {var}", data=df).fit()
        linhas.append({"preditor": var, "coef": m.params[var],
                       "pvalor": m.pvalues[var], "r2": m.rsquared})
    return pd.DataFrame(linhas).set_index("preditor").sort_values("pvalor")


def regressao_multipla(df: pd.DataFrame, preditores: list[str]):
    return smf.ols("crim ~ " + " + ".join(preditores), data=df).fit()


def regressoes_polinomiais(df: pd.DataFrame, preditores: list[str]) -> pd.DataFrame:
    """Ajusta crim ~ X + X² + X³ para cada preditor."""
    linhas = []
    for var in preditores:
        if df[var].nunique() <= 2:
            # Variável binária (chas): X² = X³ = X, o termo polinomial não existe.
            linhas.append({"preditor": var, "p_linear": None,
                           "p_quadratico": None, "p_cubico": None, "r2": None})
            continue
        m = smf.ols(f"crim ~ {var} + I({var}**2) + I({var}**3)", data=df).fit()
        linhas.append({
            "preditor": var,
            "p_linear": m.pvalues[var],
            "p_quadratico": m.pvalues[f"I({var} ** 2)"],
            "p_cubico": m.pvalues[f"I({var} ** 3)"],
            "r2": m.rsquared,
        })
    out = pd.DataFrame(linhas).set_index("preditor")
    out["nao_linear"] = (out["p_quadratico"] < NIVEL) | (out["p_cubico"] < NIVEL)
    return out


def plot_coeficientes(comp: pd.DataFrame) -> None:
    """Coeficiente da regressão simples (x) contra o da múltipla (y)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(comp["coef_simples"], comp["coef_multiplo"])
    for nome, row in comp.iterrows():
        ax.annotate(nome, (row["coef_simples"], row["coef_multiplo"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=9)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Coeficiente na regressão simples")
    ax.set_ylabel("Coeficiente na regressão múltipla")
    ax.set_title("(c) Coeficientes: regressão simples × múltipla")
    fig.tight_layout()
    save_figure(fig, "q3c_boston_coeficientes")


def plot_dispersoes(df: pd.DataFrame, variaveis: list[str]) -> None:
    fig, axes = plt.subplots(1, len(variaveis), figsize=(5 * len(variaveis), 4))
    for ax, var in zip(axes, variaveis):
        ax.scatter(df[var], df["crim"], s=10, alpha=0.6)
        ax.set_xlabel(var)
        ax.set_ylabel("crim")
        ax.set_title(f"crim × {var}")
    fig.tight_layout()
    save_figure(fig, "q3a_boston_dispersoes")


def run() -> dict:
    header("QUESTÃO 3 - Regressões para crim (Boston)")
    boston = load_boston()
    preditores = [c for c in boston.columns if c != "crim"]
    print(f"Observações: {len(boston)} | Preditores: {', '.join(preditores)}")

    # (a) Regressões simples
    simples = regressoes_simples(boston, preditores)
    save_table(simples, "q3a_boston_simples")
    print("\n(a) Regressões simples (ordenadas por p-valor):")
    print(simples.round(4).to_string())
    print("Não significativos a 5%:",
          ", ".join(simples.query("pvalor >= @NIVEL").index) or "nenhum")
    plot_dispersoes(boston, ["lstat", "rad", "dis", "medv"])

    # (b) Regressão múltipla
    multiplo = regressao_multipla(boston, preditores)
    print("\n(b) Regressão múltipla")
    print(multiplo.summary())
    coefs = pd.DataFrame({"coef": multiplo.params, "pvalor": multiplo.pvalues,
                          "t": multiplo.tvalues}).drop("Intercept").sort_values("pvalor")
    save_table(coefs, "q3b_boston_multiplo")
    print("Rejeitamos H0: beta_j = 0 (5%) para:",
          ", ".join(coefs.query("pvalor < @NIVEL").index))

    # (c) Comparação simples × múltipla
    comp = pd.DataFrame({
        "coef_simples": simples["coef"], "coef_multiplo": coefs["coef"],
        "pvalor_simples": simples["pvalor"], "pvalor_multiplo": coefs["pvalor"],
    })
    comp["sig_simples"] = comp["pvalor_simples"] < NIVEL
    comp["sig_multiplo"] = comp["pvalor_multiplo"] < NIVEL
    save_table(comp, "q3c_boston_simples_vs_multiplo")
    print("\n(c) Simples × múltipla:")
    print(comp.round(4).to_string())
    plot_coeficientes(comp)

    # (d) Não linearidade
    poli = regressoes_polinomiais(boston, preditores)
    save_table(poli, "q3d_boston_polinomial")
    print("\n(d) Modelos cúbicos:")
    print(poli.round(4).to_string())
    print("Evidência de não linearidade em:",
          ", ".join(poli.index[poli["nao_linear"]]))

    return {"simples": simples, "multiplo": multiplo, "polinomial": poli}


if __name__ == "__main__":
    from .config import ensure_dirs
    ensure_dirs()
    run()
