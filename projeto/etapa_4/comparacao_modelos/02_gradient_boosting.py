#!/usr/bin/env python3
"""
02 — Gradient Boosting (HistGradientBoostingRegressor) com Clima, Calendário e Defasagens.

Implementa o algoritmo orientado pelo Prof. Cristiano Geraldo Teixeira Silva:
1. Busca em grade de hiperparâmetros na validação temporal (2023) sem tocar no teste (2024).
2. Retreinamento no histórico 2016–2023 com transformação logarítmica do alvo (np.log/np.exp)
   para estabilizar a variância entre subsistemas de escalas distintas (SE ~ 7x N).
3. Avaliação no teste holdout de 2024 (MAE, RMSE, R², MAPE global e por subsistema).
4. Estudo de ablação (Hipóteses H1 — variáveis climáticas e H3 — defasagens e calendário).
5. Importância por permutação (Hipótese H5) e curva de dependência parcial da temperatura.
"""

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from preparacao import criar_defasagens, montar_base, separar

SAIDA = Path(__file__).resolve().parent / "resultados"
SAIDA.mkdir(exist_ok=True)

SEED = 42
ALVO = "val_cargaenergiamwmed"
SUBSISTEMAS = ["SE", "S", "NE", "N"]

CATEGORICAS = ["id_subsistema", "dia_semana", "mes"]
CLIMA = [
    "temp_max",
    "temp_media",
    "temp_min",
    "precipitacao_total",
    "radiacao_global",
    "vento_vel_media",
    "temp_media_lag_1",
    "temp_media_3d",
]
CALENDARIO = ["feriado", "dia_ano", "tendencia"]
DEFASAGENS = ["carga_lag_1", "carga_lag_7", "carga_media_7d"]
TODAS = CATEGORICAS + CLIMA + CALENDARIO + DEFASAGENS


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


def com_log(modelo):
    return TransformedTargetRegressor(regressor=modelo, func=np.log, inverse_func=np.exp)


def gradient_boosting(colunas: list[str], **params) -> Pipeline:
    cats = [c for c in colunas if c in CATEGORICAS]
    nums = [c for c in colunas if c not in CATEGORICAS]
    prep = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cats,
            ),
            ("num", "passthrough", nums),
        ],
        verbose_feature_names_out=False,
    )
    hgb = HistGradientBoostingRegressor(
        categorical_features=list(range(len(cats))) if cats else None,
        random_state=SEED,
        **params,
    )
    return Pipeline([("prep", prep), ("reg", com_log(hgb))])


def main() -> None:
    base = criar_defasagens(montar_base())
    treino, validacao, teste = separar(base, modo="2024")
    treino_final = pd.concat([treino, validacao], ignore_index=True)
    teste = teste.copy()

    print(
        f"Treino inicial (2016-2022): {len(treino)} | "
        f"Validação (2023): {len(validacao)} | "
        f"Teste final (2024): {len(teste)}"
    )

    # 1. Busca de hiperparâmetros na validação temporal (2023)
    grade = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [400, 800, 1000],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [10, 30],
    }
    resultados_grade = []
    for valores in itertools.product(*grade.values()):
        params = dict(zip(grade.keys(), valores))
        m = gradient_boosting(TODAS, **params).fit(treino[TODAS], treino[ALVO])
        pred_val = m.predict(validacao[TODAS])
        metricas_val = avaliar(validacao[ALVO], pred_val)
        resultados_grade.append({
            **params,
            "MAPE_validacao_%": metricas_val["MAPE_%"],
            "RMSE_validacao": metricas_val["RMSE"],
        })

    grade_df = pd.DataFrame(resultados_grade).sort_values("MAPE_validacao_%").reset_index(drop=True)
    grade_df.to_csv(SAIDA / "gb_busca_hiperparametros.csv", index=False)

    melhores_raw = grade_df.iloc[0][list(grade.keys())].to_dict()
    melhores = {
        k: (float(v) if k == "learning_rate" else int(v))
        for k, v in melhores_raw.items()
    }
    print(
        f"\nMelhores hiperparâmetros GB (validação 2023): {melhores} -> "
        f"MAPE val = {grade_df.iloc[0]['MAPE_validacao_%']:.2f}%"
    )

    # 2. Treinamento final (2016-2023) e avaliação no teste (2024)
    modelo_gb = gradient_boosting(TODAS, **melhores)
    modelo_gb.fit(treino_final[TODAS], treino_final[ALVO])

    teste["previsto_gb"] = modelo_gb.predict(teste[TODAS])
    metricas_teste = avaliar(teste[ALVO], teste["previsto_gb"])
    metricas_treino = avaliar(treino_final[ALVO], modelo_gb.predict(treino_final[TODAS]))
    por_sub = mape_por_subsistema(teste, "previsto_gb")

    print("\nDesempenho no TESTE (2024) — Gradient Boosting:")
    for k, v in metricas_teste.items():
        print(f"  {k:7s} {v:12.4f}" if k == "R2" else f"  {k:7s} {v:12.2f}")
    print(f"  (MAPE no treino 2016-2023: {metricas_treino['MAPE_%']:.2f}% | R² treino: {metricas_treino['R2']:.4f})")
    print("MAPE por subsistema (%):", por_sub.round(2).to_dict())

    # 3. Estudo de Ablação (H1 e H3)
    cenarios = {
        "Completo": TODAS,
        "Sem variáveis climáticas (H1)": [c for c in TODAS if c not in CLIMA],
        "Sem defasagens da carga (H3)": [c for c in TODAS if c not in DEFASAGENS],
        "Sem calendário (H3)": [c for c in TODAS if c not in CALENDARIO + ["dia_semana", "mes"]],
    }
    ablacao_dict = {}
    for nome_cen, cols in cenarios.items():
        m_cen = gradient_boosting(cols, **melhores).fit(treino_final[cols], treino_final[ALVO])
        ablacao_dict[nome_cen] = avaliar(teste[ALVO], m_cen.predict(teste[cols]))
    ablacao_df = pd.DataFrame(ablacao_dict).T
    print("\nAblação do Gradient Boosting (Teste 2024):")
    print(ablacao_df.round(3).to_string())
    ablacao_df.round(4).to_csv(SAIDA / "gb_ablacao.csv")

    # 4. Importância por Permutação (H5)
    imp = permutation_importance(
        modelo_gb,
        teste[TODAS],
        teste[ALVO],
        n_repeats=10,
        random_state=SEED,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
    )
    importancia = pd.Series(imp.importances_mean * 100, index=TODAS).sort_values(ascending=False)
    print("\nImportância por permutação — GB (aumento do MAPE em p.p.):")
    print(importancia.round(3).to_string())
    importancia.to_csv(SAIDA / "gb_importancia.csv", header=["aumento_mape_pp"])

    # 5. Gráficos específicos do Gradient Boosting
    # 5a. Série 2024 Real x Previsto por subsistema
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, sub in zip(axs.ravel(), SUBSISTEMAS):
        t = teste[teste["id_subsistema"] == sub]
        ax.plot(t["event_date"], t[ALVO], lw=0.9, color="#1f2937", label="Real")
        ax.plot(t["event_date"], t["previsto_gb"], lw=0.9, color="#059669", label="Gradient Boosting")
        ax.set_title(f"Subsistema {sub} — MAPE {por_sub[sub]:.2f}%".replace(".", ","))
        ax.set_ylabel("MWmed")
        ax.grid(alpha=0.25)
    axs[0, 0].legend()
    fig.suptitle("Gradient Boosting — Carga diária real x prevista em 2024", fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SAIDA / "gb_series_2024.png", dpi=150)
    plt.close(fig)

    # 5b. Importância por permutação (excluindo id_subsistema que representa apenas a escala base)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    imp_plot = importancia.drop("id_subsistema").head(12)[::-1]
    ax.barh(imp_plot.index, imp_plot.values, color="#059669")
    for i, v in enumerate(imp_plot.values):
        ax.text(max(v, 0) + 0.03, i, f"{v:.2f}".replace(".", ",") + " p.p.", va="center", fontsize=8.5)
    ax.set_xlim(0, imp_plot.max() * 1.18)
    ax.set(
        xlabel="Aumento do MAPE ao embaralhar a variável (pontos percentuais)",
        title="Gradient Boosting — Importância das variáveis no teste 2024 (H5)",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "gb_importancia.png", dpi=150)
    plt.close(fig)

    # 5c. Curva de dependência parcial manual da temperatura por subsistema
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    grade_t = np.linspace(14, 32, 37)
    cores_sub = {"SE": "#2563eb", "S": "#7c3aed", "NE": "#d97706", "N": "#059669"}
    for sub in SUBSISTEMAS:
        amostra = teste[(teste["id_subsistema"] == sub) & (teste["dia_semana"] < 5)]
        gt = grade_t[grade_t >= 22] if sub in ("NE", "N") else grade_t
        resp = []
        for tv in gt:
            X_sim = amostra[TODAS].copy()
            X_sim[["temp_media", "temp_media_lag_1", "temp_media_3d"]] = tv
            X_sim["temp_max"] = tv + 5.0
            X_sim["temp_min"] = tv - 5.0
            resp.append(modelo_gb.predict(X_sim).mean())
        resp_arr = np.array(resp)
        ax.plot(gt, (resp_arr / resp_arr[0] - 1) * 100, lw=2, color=cores_sub[sub], label=sub)
    ax.axhline(0, color="#6b7280", lw=0.8, ls=":")
    ax.set(
        xlabel="Temperatura média do subsistema (°C)",
        ylabel="Variação da carga prevista (%)",
        title="Resposta não linear da carga à temperatura aprendida pelo Gradient Boosting",
    )
    ax.legend(title="Subsistema")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "gb_resposta_temperatura.png", dpi=150)
    plt.close(fig)

    (SAIDA / "metricas_gradient_boosting.json").write_text(
        json.dumps(
            {
                "modelo": "Gradient Boosting (HistGradientBoostingRegressor + log)",
                "melhores_hiperparametros": melhores,
                "mape_validacao_2023_%": float(grade_df.iloc[0]["MAPE_validacao_%"]),
                "teste": metricas_teste,
                "treino": metricas_treino,
                "mape_subsistema": por_sub.round(4).to_dict(),
                "ablacao": ablacao_df.round(4).to_dict(orient="index"),
                "importancia_permutacao_pp": importancia.round(4).to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    teste.to_csv(SAIDA / "gb_previsoes_2024.csv", index=False)
    print(f"\nResultados salvos em {SAIDA}")


if __name__ == "__main__":
    main()
