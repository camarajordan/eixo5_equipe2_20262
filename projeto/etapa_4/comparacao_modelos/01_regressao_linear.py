#!/usr/bin/env python3
"""
01 — Baseline de Regressão Linear Diária com Clima e Calendário.

Segue a referência do orientador (Prof. Cristiano Geraldo Teixeira Silva):
- Treino cronológico: 2016–2023 | Teste holdout: 2024.
- Pipeline sem vazamento temporal: SimpleImputer(median) + StandardScaler nas numéricas
  e OneHotEncoder(drop='first') nas categóricas, ajustados exclusivamente no treino.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preparacao import montar_base, separar

SAIDA = Path(__file__).resolve().parent / "resultados"
SAIDA.mkdir(exist_ok=True)

ALVO = "val_cargaenergiamwmed"
NUMERICAS = [
    "temp_max",
    "temp_media",
    "temp_min",
    "precipitacao_total",
    "radiacao_global",
    "vento_vel_media",
    "mes",
    "feriado",
    "tendencia",
]
CATEGORICAS = ["id_subsistema", "dia_semana"]
SUBSISTEMAS = ["SE", "S", "NE", "N"]


def avaliar(y_real: pd.Series | np.ndarray, y_prev: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_real, y_prev)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_prev))),
        "R2": float(r2_score(y_real, y_prev)),
        "MAPE_%": float(mean_absolute_percentage_error(y_real, y_prev) * 100),
    }


def mape_por_subsistema(df: pd.DataFrame, col_prev: str) -> pd.Series:
    valores = {
        sub: float(mean_absolute_percentage_error(g[ALVO], g[col_prev]) * 100)
        for sub, g in df.groupby("id_subsistema")
    }
    return pd.Series(valores).loc[SUBSISTEMAS]


def main() -> None:
    base = montar_base()
    treino, validacao, teste = separar(base, modo="2024")
    treino_completo = pd.concat([treino, validacao], ignore_index=True)
    teste = teste.copy()

    print(
        f"Treino: {len(treino_completo)} linhas (2016-2023) | "
        f"Teste: {len(teste)} linhas (2024)"
    )

    X_tr, y_tr = treino_completo[NUMERICAS + CATEGORICAS], treino_completo[ALVO]
    X_te, y_te = teste[NUMERICAS + CATEGORICAS], teste[ALVO]

    modelo = Pipeline([
        (
            "prep",
            ColumnTransformer([
                (
                    "num",
                    Pipeline([
                        ("imp", SimpleImputer(strategy="median")),
                        ("esc", StandardScaler()),
                    ]),
                    NUMERICAS,
                ),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAS),
            ]),
        ),
        ("reg", LinearRegression()),
    ])
    modelo.fit(X_tr, y_tr)

    teste["previsto"] = modelo.predict(X_te)
    metricas = avaliar(y_te, teste["previsto"])
    metricas_treino = avaliar(y_tr, modelo.predict(X_tr))
    por_sub = mape_por_subsistema(teste, "previsto")

    print("\nDesempenho no TESTE (2024) — Regressão Linear:")
    for k, v in metricas.items():
        print(f"  {k:7s} {v:12.4f}" if k == "R2" else f"  {k:7s} {v:12.2f}")
    print(f"  (R² no treino: {metricas_treino['R2']:.4f})")
    print("MAPE por subsistema (%):", por_sub.round(2).to_dict())

    nomes = modelo.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(modelo.named_steps["reg"].coef_, index=nomes).sort_values(
        key=abs, ascending=False
    )
    print("\nCoeficientes (MWmed por +1 desvio-padrão; categorias vs. referência N / segunda):")
    print(coefs.round(1).to_string())

    # Gráfico 1: Série diária real x prevista em 2024 por subsistema
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, sub in zip(axs.ravel(), SUBSISTEMAS):
        t = teste[teste["id_subsistema"] == sub]
        ax.plot(t["event_date"], t[ALVO], lw=0.9, color="#1f2937", label="Real")
        ax.plot(t["event_date"], t["previsto"], lw=0.9, color="#d97706", label="Regressão Linear")
        ax.set_title(f"Subsistema {sub} — MAPE {por_sub[sub]:.2f}%".replace(".", ","))
        ax.set_ylabel("MWmed")
        ax.grid(alpha=0.25)
    axs[0, 0].legend()
    fig.suptitle("Regressão Linear — Carga diária real x prevista em 2024", fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SAIDA / "rl_series_2024.png", dpi=150)
    plt.close(fig)

    # Gráfico 2: Resíduos x Temperatura média no Subsistema SE
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    t_se = teste[teste["id_subsistema"] == "SE"].dropna(subset=["temp_media"])
    ax.scatter(
        t_se["temp_media"],
        t_se[ALVO] - t_se["previsto"],
        s=14,
        alpha=0.65,
        color="#2563eb",
    )
    ax.axhline(0, color="#111827", lw=1, ls="--")
    ax.set(
        xlabel="Temperatura média do dia (°C)",
        ylabel="Resíduo (Real − Previsto, MWmed)",
        title="Subsistema SE — Resíduos da Regressão Linear x Temperatura (2024)",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "rl_residuo_temperatura.png", dpi=150)
    plt.close(fig)

    (SAIDA / "metricas_regressao_linear.json").write_text(
        json.dumps(
            {
                "modelo": "Regressão Linear (clima + calendário, sem lags)",
                "teste": metricas,
                "treino": metricas_treino,
                "mape_subsistema": por_sub.round(4).to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    coefs.to_csv(SAIDA / "rl_coeficientes.csv", header=["coeficiente"])
    teste.to_csv(SAIDA / "rl_previsoes_2024.csv", index=False)
    print(f"\nResultados salvos em {SAIDA}")


if __name__ == "__main__":
    main()
