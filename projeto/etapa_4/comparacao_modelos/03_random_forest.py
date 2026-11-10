#!/usr/bin/env python3
"""
03 — Random Forest (3º Algoritmo) e Consolidação Comparativa dos Modelos.

Implementa o RandomForestRegressor (ensemble paralelo via Bagging + amostragem aleatória
de atributos por nó) com o mesmo protocolo experimental rigoroso:
1. Busca em grade de hiperparâmetros na validação temporal (2023):
   n_estimators, max_features e min_samples_leaf (controle de variância/overfitting).
2. Retreinamento em 2016–2023 com SimpleImputer(median) restrito ao treino + transformação log
   do alvo (np.log/np.exp) e avaliação no teste holdout de 2024.
3. Estudo de ablação (H1 e H3) e importância por permutação (H5) do Random Forest.
4. Consolidação comparativa completa (Regressão Linear sem lags, Regressão Linear com lags
   em log-log, Baseline Ingênuo, Random Forest e Gradient Boosting) em tabelas e gráficos.

Pré-requisito: executar antes 01_regressao_linear.py e 02_gradient_boosting.py, pois este
script lê os resultados deles em resultados/ para montar a comparação consolidada.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

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


def random_forest(colunas: list[str], **params) -> Pipeline:
    cats = [c for c in colunas if c in CATEGORICAS]
    nums = [c for c in colunas if c not in CATEGORICAS]
    prep = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cats,
            ),
            ("num", SimpleImputer(strategy="median"), nums),
        ],
        verbose_feature_names_out=False,
    )
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    return Pipeline([("prep", prep), ("reg", com_log(rf))])


def regressao_linear_com_lags(colunas: list[str]) -> Pipeline:
    """
    Regressão Linear log-log: como o alvo é modelado em log, as defasagens da carga
    também entram em log. Com as defasagens em MW, a relação log(carga) x carga_lag
    ficaria mal especificada (escalas de SE ~ 45 mil MW e N ~ 7,5 mil MW).
    """
    cats = [c for c in colunas if c in CATEGORICAS]
    lags = [c for c in colunas if c in DEFASAGENS]
    nums = [c for c in colunas if c not in CATEGORICAS and c not in DEFASAGENS]
    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("esc", StandardScaler()),
            ]),
            nums,
        ),
        (
            "lag_log",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                ("esc", StandardScaler()),
            ]),
            lags,
        ),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cats),
    ])
    return Pipeline([("prep", prep), ("reg", com_log(LinearRegression()))])


def _gerar_markdown_resumo(
    tabela: pd.DataFrame,
    por_sub: pd.DataFrame,
    ablacao_gb: pd.DataFrame | None,
    ablacao_rf: pd.DataFrame,
    melhores_gb: dict | None,
    melhores_rf: dict,
) -> str:
    linhas = [
        "# Resumo Comparativo dos Modelos — Teste em 2024",
        "",
        "Resultados obtidos no período de teste (`2024`, `1.464` registros diários nos 4 subsistemas elétricos: `SE`, `S`, `NE` e `N`), após treinamento com a série histórica de `2016` a `2023`.",
        "",
    ]
    if melhores_gb:
        linhas.append(
            f"- **Configuração selecionada para o Gradient Boosting (validação 2023):** `{melhores_gb}`"
        )
    linhas.extend([
        f"- **Configuração selecionada para o Random Forest (validação 2023):** `{melhores_rf}`",
        "",
        "### Legenda das Métricas Avaliadas",
        "- **MAE (Erro Médio Absoluto, em MWmed):** média simples de quantos Megawatts o modelo errou por dia (quanto menor, melhor).",
        "- **RMSE (Raiz do Erro Quadrático Médio, em MWmed):** similar ao MAE, mas penaliza com maior peso os erros grandes em dias de pico (quanto menor, melhor).",
        "- **R² (Coeficiente de Determinação):** varia até `1,00` e indica a proporção da variação real explicada pelo modelo (quanto mais próximo de `1,00`, melhor).",
        "- **MAPE (Erro Percentual Absoluto Médio, em %):** erro médio em porcentagem da carga real (quanto menor, melhor).",
        "",
        "---",
        "",
        "## 1. Métricas Globais no Período de Teste (2024)",
        "",
        "| Modelo | MAE (MWmed) | RMSE (MWmed) | R² | MAPE (%) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for idx, row in tabela.sort_values("MAPE_%").iterrows():
        linhas.append(
            f"| {idx} | {row['MAE']:.2f} | {row['RMSE']:.2f} | {row['R2']:.4f} | **{row['MAPE_%']:.2f}%** |"
        )

    linhas.extend([
        "",
        "## 2. Erro Percentual (MAPE) por Subsistema em 2024",
        "",
        "| Subsistema | " + " | ".join(por_sub.columns) + " |",
        "| :--- | " + " | ".join(["---:"] * len(por_sub.columns)) + " |",
    ])
    for sub, row in por_sub.iterrows():
        vals = " | ".join(f"{v:.2f}%" for v in row.values)
        linhas.append(f"| **{sub}** | {vals} |")

    if ablacao_gb is not None:
        linhas.extend([
            "",
            "## 3. Avaliação do Impacto de Cada Grupo de Variáveis (Ablação)",
            "",
            "| Cenário Avaliado | Gradient Boosting — MAPE (%) | Random Forest — MAPE (%) |",
            "| :--- | ---: | ---: |",
        ])
        for cen in ablacao_rf.index:
            gb_val = f"{ablacao_gb.loc[cen, 'MAPE_%']:.2f}%" if cen in ablacao_gb.index else "-"
            rf_val = f"{ablacao_rf.loc[cen, 'MAPE_%']:.2f}%"
            linhas.append(f"| {cen} | {gb_val} | {rf_val} |")

    return "\n".join(linhas) + "\n"


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

    # 1. Busca de hiperparâmetros do Random Forest na validação (2023)
    grade_rf = {
        "n_estimators": [200, 400],
        "max_features": [0.5, 0.7, 1.0],
        "min_samples_leaf": [1, 3, 5],
    }
    resultados_grade = []
    for valores in itertools.product(*grade_rf.values()):
        params = dict(zip(grade_rf.keys(), valores))
        m = random_forest(TODAS, **params).fit(treino[TODAS], treino[ALVO])
        metricas_val = avaliar(validacao[ALVO], m.predict(validacao[TODAS]))
        resultados_grade.append({
            **params,
            "MAPE_validacao_%": metricas_val["MAPE_%"],
            "RMSE_validacao": metricas_val["RMSE"],
        })

    grade_df = pd.DataFrame(resultados_grade).sort_values("MAPE_validacao_%").reset_index(drop=True)
    grade_df.to_csv(SAIDA / "rf_busca_hiperparametros.csv", index=False)

    melhores_raw = grade_df.iloc[0][list(grade_rf.keys())].to_dict()
    melhores_rf = {
        "n_estimators": int(melhores_raw["n_estimators"]),
        "max_features": float(melhores_raw["max_features"]),
        "min_samples_leaf": int(melhores_raw["min_samples_leaf"]),
    }
    print(
        f"\nMelhores hiperparâmetros RF (validação 2023): {melhores_rf} -> "
        f"MAPE val = {grade_df.iloc[0]['MAPE_validacao_%']:.2f}%"
    )

    # 2. Treinamento final (2016-2023) e teste (2024) do Random Forest + Regressão Linear c/ lags
    modelo_rf = random_forest(TODAS, **melhores_rf)
    modelo_rf.fit(treino_final[TODAS], treino_final[ALVO])
    teste["previsto_rf"] = modelo_rf.predict(teste[TODAS])

    metricas_rf_teste = avaliar(teste[ALVO], teste["previsto_rf"])
    metricas_rf_treino = avaliar(treino_final[ALVO], modelo_rf.predict(treino_final[TODAS]))
    por_sub_rf = mape_por_subsistema(teste, "previsto_rf")

    print("\nDesempenho no TESTE (2024) — Random Forest:")
    for k, v in metricas_rf_teste.items():
        print(f"  {k:7s} {v:12.4f}" if k == "R2" else f"  {k:7s} {v:12.2f}")
    print(f"  (MAPE no treino 2016-2023: {metricas_rf_treino['MAPE_%']:.2f}% | R² treino: {metricas_rf_treino['R2']:.4f})")
    print("MAPE por subsistema (%):", por_sub_rf.round(2).to_dict())

    modelo_rl_lags = regressao_linear_com_lags(TODAS).fit(treino_final[TODAS], treino_final[ALVO])
    teste["previsto_rl_lags"] = modelo_rl_lags.predict(teste[TODAS])
    metricas_rl_lags = avaliar(teste[ALVO], teste["previsto_rl_lags"])
    por_sub_rl_lags = mape_por_subsistema(teste, "previsto_rl_lags")

    # 3. Ablação do Random Forest (H1 e H3)
    cenarios = {
        "Completo": TODAS,
        "Sem variáveis climáticas (H1)": [c for c in TODAS if c not in CLIMA],
        "Sem defasagens da carga (H3)": [c for c in TODAS if c not in DEFASAGENS],
        "Sem calendário (H3)": [c for c in TODAS if c not in CALENDARIO + ["dia_semana", "mes"]],
    }
    ablacao_rf_dict = {}
    for nome_cen, cols in cenarios.items():
        m_cen = random_forest(cols, **melhores_rf).fit(treino_final[cols], treino_final[ALVO])
        ablacao_rf_dict[nome_cen] = avaliar(teste[ALVO], m_cen.predict(teste[cols]))
    ablacao_rf_df = pd.DataFrame(ablacao_rf_dict).T
    print("\nAblação do Random Forest (Teste 2024):")
    print(ablacao_rf_df.round(3).to_string())
    ablacao_rf_df.round(4).to_csv(SAIDA / "rf_ablacao.csv")

    # 4. Importância por Permutação do Random Forest (H5)
    imp_rf = permutation_importance(
        modelo_rf,
        teste[TODAS],
        teste[ALVO],
        n_repeats=10,
        random_state=SEED,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
    )
    importancia_rf = pd.Series(imp_rf.importances_mean * 100, index=TODAS).sort_values(ascending=False)
    importancia_rf.to_csv(SAIDA / "rf_importancia.csv", header=["aumento_mape_pp"])

    # Gráficos individuais do Random Forest
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, sub in zip(axs.ravel(), SUBSISTEMAS):
        t = teste[teste["id_subsistema"] == sub]
        ax.plot(t["event_date"], t[ALVO], lw=0.9, color="#1f2937", label="Real")
        ax.plot(t["event_date"], t["previsto_rf"], lw=0.9, color="#2563eb", label="Random Forest")
        ax.set_title(f"Subsistema {sub} — MAPE {por_sub_rf[sub]:.2f}%".replace(".", ","))
        ax.set_ylabel("MWmed")
        ax.grid(alpha=0.25)
    axs[0, 0].legend()
    fig.suptitle("Random Forest — Carga diária real x prevista em 2024", fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SAIDA / "rf_series_2024.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    imp_plot = importancia_rf.drop("id_subsistema").head(12)[::-1]
    ax.barh(imp_plot.index, imp_plot.values, color="#2563eb")
    for i, v in enumerate(imp_plot.values):
        ax.text(max(v, 0) + 0.03, i, f"{v:.2f}".replace(".", ",") + " p.p.", va="center", fontsize=8.5)
    ax.set_xlim(0, imp_plot.max() * 1.18)
    ax.set(
        xlabel="Aumento do MAPE ao embaralhar a variável (pontos percentuais)",
        title="Random Forest — Importância das variáveis no teste 2024 (H5)",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "rf_importancia.png", dpi=150)
    plt.close(fig)

    (SAIDA / "metricas_random_forest.json").write_text(
        json.dumps(
            {
                "modelo": "Random Forest (RandomForestRegressor + log)",
                "melhores_hiperparametros": melhores_rf,
                "mape_validacao_2023_%": float(grade_df.iloc[0]["MAPE_validacao_%"]),
                "teste": metricas_rf_teste,
                "treino": metricas_rf_treino,
                "mape_subsistema": por_sub_rf.round(4).to_dict(),
                "ablacao": ablacao_rf_df.round(4).to_dict(orient="index"),
                "importancia_permutacao_pp": importancia_rf.round(4).to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    teste.to_csv(SAIDA / "rf_previsoes_2024.csv", index=False)

    # 5. Consolidação comparativa com Regressão Linear (script 01) e Gradient Boosting (script 02)
    comparacao: dict[str, dict[str, float]] = {}
    rl_json = SAIDA / "metricas_regressao_linear.json"
    gb_json = SAIDA / "metricas_gradient_boosting.json"
    rl_prev = (
        pd.read_csv(SAIDA / "rl_previsoes_2024.csv", parse_dates=["event_date"])
        if (SAIDA / "rl_previsoes_2024.csv").exists()
        else None
    )
    gb_prev = (
        pd.read_csv(SAIDA / "gb_previsoes_2024.csv", parse_dates=["event_date"])
        if (SAIDA / "gb_previsoes_2024.csv").exists()
        else None
    )

    if rl_json.exists():
        comparacao["Regressão Linear (clima, sem lags)"] = json.loads(
            rl_json.read_text(encoding="utf-8")
        )["teste"]
    comparacao["Regressão Linear + Lags (log)"] = metricas_rl_lags
    comparacao["Baseline Ingênuo (carga_lag_7)"] = avaliar(teste[ALVO], teste["carga_lag_7"])
    comparacao["Random Forest"] = metricas_rf_teste
    melhores_gb = None
    ablacao_gb_df = None
    if gb_json.exists():
        gb_info = json.loads(gb_json.read_text(encoding="utf-8"))
        comparacao["Gradient Boosting"] = gb_info["teste"]
        melhores_gb = gb_info.get("melhores_hiperparametros")
        ablacao_gb_df = pd.DataFrame(gb_info["ablacao"]).T

    tabela = pd.DataFrame(comparacao).T.sort_values("MAPE_%", ascending=False)
    print("\n=== COMPARAÇÃO CONSOLIDADA NO TESTE (2024) ===")
    print(tabela.round(3).to_string())
    tabela.round(4).to_csv(SAIDA / "comparacao_modelos.csv")

    colunas_sub: dict[str, pd.Series] = {}
    if rl_prev is not None:
        colunas_sub["Regressão Linear (sem lags)"] = mape_por_subsistema(rl_prev, "previsto")
    colunas_sub["Regressão Linear + Lags"] = por_sub_rl_lags
    colunas_sub["Ingênuo (lag 7)"] = mape_por_subsistema(teste, "carga_lag_7")
    colunas_sub["Random Forest"] = por_sub_rf
    if gb_prev is not None:
        colunas_sub["Gradient Boosting"] = mape_por_subsistema(gb_prev, "previsto_gb")
    por_sub_df = pd.DataFrame(colunas_sub).loc[SUBSISTEMAS]
    print("\n=== MAPE (%) POR SUBSISTEMA EM 2024 ===")
    print(por_sub_df.round(2).to_string())
    por_sub_df.round(3).to_csv(SAIDA / "mape_por_subsistema.csv")

    # Gráfico Comparativo 1: MAPE global de todos os modelos
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ordem = tabela["MAPE_%"].sort_values(ascending=True)[::-1]
    cores_bar = [
        "#059669" if "Gradient Boosting" in m
        else "#2563eb" if "Random Forest" in m
        else "#6b7280" if "Ingênuo" in m
        else "#d97706"
        for m in ordem.index
    ]
    ax.barh(ordem.index, ordem.values, color=cores_bar)
    for i, v in enumerate(ordem.values):
        ax.text(v + 0.12, i, f"{v:.2f}%".replace(".", ","), va="center", fontweight="bold", fontsize=9)
    ax.set_xlim(0, ordem.max() * 1.16)
    ax.set(
        xlabel="MAPE no teste holdout 2024 (%) — menor é melhor",
        title="Comparação de Modelos — Previsão Diária de Carga Elétrica (2024)",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "comparacao_mape.png", dpi=150)
    plt.close(fig)

    # Gráfico Comparativo 2: MAPE por Subsistema (RL vs RF vs GB)
    cols_principais = [
        c for c in ["Regressão Linear (sem lags)", "Random Forest", "Gradient Boosting"]
        if c in por_sub_df.columns
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(SUBSISTEMAS))
    largura = 0.25
    paleta = {
        "Regressão Linear (sem lags)": "#d97706",
        "Random Forest": "#2563eb",
        "Gradient Boosting": "#059669",
    }
    for idx_c, col in enumerate(cols_principais):
        desloc = (idx_c - (len(cols_principais) - 1) / 2) * largura
        barras = ax.bar(x + desloc, por_sub_df[col].values, width=largura, label=col, color=paleta[col])
        for b in barras:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.15,
                f"{h:.1f}%".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Subsistema {s}" for s in SUBSISTEMAS])
    ax.set_ylim(0, por_sub_df[cols_principais].to_numpy().max() * 1.18)
    ax.set(
        ylabel="MAPE (%)",
        title="Erro Percentual (MAPE) por Subsistema em 2024 — Linear vs. Árvores",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SAIDA / "mape_por_subsistema.png", dpi=150)
    plt.close(fig)

    # Gráfico Comparativo 3: Zoom de 6 semanas no SE (fev-mar/2024 — Carnaval + ondas de calor)
    t_se = teste[
        (teste["id_subsistema"] == "SE") & teste["event_date"].between("2024-02-01", "2024-03-15")
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(t_se["event_date"], t_se[ALVO], color="#111827", lw=1.8, label="Real (ONS)")
    if rl_prev is not None:
        r_se = rl_prev[
            (rl_prev["id_subsistema"] == "SE")
            & rl_prev["event_date"].between("2024-02-01", "2024-03-15")
        ]
        ax.plot(
            r_se["event_date"],
            r_se["previsto"],
            "--",
            color="#d97706",
            lw=1.2,
            label="Regressão Linear",
        )
    ax.plot(
        t_se["event_date"],
        t_se["previsto_rf"],
        "-.",
        color="#2563eb",
        lw=1.3,
        label="Random Forest",
    )
    if gb_prev is not None:
        g_se = gb_prev[
            (gb_prev["id_subsistema"] == "SE")
            & gb_prev["event_date"].between("2024-02-01", "2024-03-15")
        ]
        ax.plot(
            g_se["event_date"],
            g_se["previsto_gb"],
            "-",
            color="#059669",
            lw=1.4,
            label="Gradient Boosting",
        )
    for d in t_se.loc[t_se["feriado"] == 1, "event_date"]:
        ax.axvline(d, color="#6b7280", ls=":", lw=1.1)
    ax.set(
        ylabel="Carga (MWmed)",
        title="Subsistema SE — Fev a Mar/2024 (Carnaval em linhas pontilhadas e ondas de calor)",
    )
    ax.legend(ncol=2, fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SAIDA / "zoom_se_fev_mar_2024.png", dpi=150)
    plt.close(fig)

    # Salva resumo consolidado em JSON e Markdown (versionáveis no Git)
    (SAIDA / "resumo_comparativo.json").write_text(
        json.dumps(
            {
                "comparacao_global_2024": tabela.round(4).to_dict(orient="index"),
                "mape_por_subsistema_2024": por_sub_df.round(4).to_dict(orient="index"),
                "hiperparametros": {
                    "gradient_boosting": melhores_gb,
                    "random_forest": melhores_rf,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_resumo = _gerar_markdown_resumo(
        tabela, por_sub_df, ablacao_gb_df, ablacao_rf_df, melhores_gb, melhores_rf
    )
    (SAIDA / "resumo_comparativo.md").write_text(md_resumo, encoding="utf-8")
    print(f"\nConsolidação concluída e salva em {SAIDA}")


if __name__ == "__main__":
    main()
