"""Testes dos métodos de intervalo de previsão em dados simulados.

Em um modelo linear com erro normal, um intervalo de previsão de 95% deve
cobrir cerca de 95% das novas observações. O bootstrap que só reamostra os
coeficientes fica muito abaixo disso, e é exatamente esse o erro corrigido.
"""
import numpy as np
import pytest

from src.prediction_intervals import (
    avaliar,
    bootstrap_so_coeficientes,
    intervalo_bootstrap,
    intervalo_conformal,
    intervalo_tradicional,
)


@pytest.fixture(scope="module")
def dados():
    rng = np.random.default_rng(0)

    def gerar(n):
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
        y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(scale=2.0, size=n)
        return X, y

    return gerar(400), gerar(400), gerar(2000)


def test_tradicional_cobre_95(dados):
    (X_tr, y_tr), _, (X_te, y_te) = dados
    _, lo, up = intervalo_tradicional(X_tr, y_tr, X_te)
    assert avaliar(y_te, lo, up)["cobertura"] == pytest.approx(0.95, abs=0.02)


def test_bootstrap_corrigido_cobre_95(dados):
    (X_tr, y_tr), _, (X_te, y_te) = dados
    _, lo, up = intervalo_bootstrap(X_tr, y_tr, X_te, B=500)
    assert avaliar(y_te, lo, up)["cobertura"] == pytest.approx(0.95, abs=0.02)


def test_conformal_cobre_95(dados):
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = dados
    _, lo, up, _ = intervalo_conformal(X_tr, y_tr, X_cal, y_cal, X_te)
    assert avaliar(y_te, lo, up)["cobertura"] == pytest.approx(0.95, abs=0.02)


def test_bootstrap_original_subcobre(dados):
    (X_tr, y_tr), _, (X_te, y_te) = dados
    _, lo, up = bootstrap_so_coeficientes(X_tr, y_tr, X_te, B=500)
    assert avaliar(y_te, lo, up)["cobertura"] < 0.5
