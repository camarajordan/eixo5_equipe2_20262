import awswrangler as wr
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

DATABASE = "db_energia_clima_puc"
S3_OUTPUT = "s3://projeto-puc-energia-clima-2026-689798927674/athena-results/"

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================

print("Carregando dados do Athena...")

query = """
SELECT
    id_subsistema,
    nom_subsistema,
    event_date,
    val_cargaenergiamwmed
FROM gold_fato_carga
ORDER BY id_subsistema, event_date
"""

df = wr.athena.read_sql_query(
    sql=query,
    database=DATABASE,
    s3_output=S3_OUTPUT,
    ctas_approach=False
)

df["event_date"] = pd.to_datetime(df["event_date"])

# ==========================================
# 3. AGREGAÇÃO MENSAL
# ==========================================

df["referencia"] = df["event_date"].dt.to_period("M")

df_mensal = (
    df.groupby(["id_subsistema", "nom_subsistema", "referencia"])
    ["val_cargaenergiamwmed"]
    .mean()
    .reset_index()
)

df_mensal = df_mensal.rename(
    columns={"val_cargaenergiamwmed": "carga_media"}
)

df_mensal["data"] = df_mensal["referencia"].dt.to_timestamp()
df_mensal["mes"] = df_mensal["data"].dt.month
df_mensal["ano"] = df_mensal["data"].dt.year

print(f"Total de registros mensais: {len(df_mensal)}")

# ==========================================
# 4. FUNÇÕES DE AVALIAÇÃO
# ==========================================

def calcular_metricas(real, previsto):
    erro = real - previsto

    mae = np.mean(np.abs(erro))
    rmse = np.sqrt(np.mean(erro ** 2))
    mape = np.mean(np.abs(erro / real)) * 100

    return mae, rmse, mape


# ==========================================
# 5. TREINAMENTO E TESTE POR SUBSISTEMA
# ==========================================

resultados = []
previsoes_teste = []
previsoes_2025 = []

for subsistema, dados in df_mensal.groupby("nom_subsistema"):

    dados = dados.sort_values("data").copy()
    dados = dados.reset_index(drop=True)

    # Tendência temporal contínua
    dados["tendencia"] = np.arange(len(dados))

    # Variáveis indicadoras dos meses
    meses = pd.get_dummies(
        dados["mes"],
        prefix="mes",
        drop_first=True,
        dtype=int
    )

    X = pd.concat(
        [dados[["tendencia"]], meses],
        axis=1
    )

    # Divisão temporal: treino até 2022, teste 2023-2024
    treino = dados["ano"] <= 2022
    teste = dados["ano"].between(2023, 2024)

    X_treino = X.loc[treino].to_numpy(dtype=float)
    X_teste = X.loc[teste].to_numpy(dtype=float)

    y_treino = dados.loc[treino, "carga_media"].to_numpy()
    y_teste = dados.loc[teste, "carga_media"].to_numpy()

    # Regressão linear por mínimos quadrados
    X_treino = np.column_stack([np.ones(len(X_treino)), X_treino])
    X_teste = np.column_stack([np.ones(len(X_teste)), X_teste])

    coeficientes = np.linalg.lstsq(
        X_treino,
        y_treino,
        rcond=None
    )[0]

    previsto = X_teste @ coeficientes

    mae, rmse, mape = calcular_metricas(y_teste, previsto)

    # Modelo de referência: mesmo mês do ano anterior
    dados["previsao_referencia"] = (
        dados.groupby("mes")["carga_media"].shift(1)
    )

    y_referencia = dados.loc[teste, "previsao_referencia"].to_numpy()

    mae_ref, rmse_ref, mape_ref = calcular_metricas(
        y_teste,
        y_referencia
    )

    resultados.append({
        "subsistema": subsistema,
        "MAE_modelo": mae,
        "RMSE_modelo": rmse,
        "MAPE_modelo_%": mape,
        "MAE_referencia": mae_ref,
        "RMSE_referencia": rmse_ref,
        "MAPE_referencia_%": mape_ref
    })

    # Armazenar previsões do período de teste
    teste_resultado = dados.loc[teste, [
        "nom_subsistema", "data", "carga_media"
    ]].copy()

    teste_resultado["previsao_modelo"] = previsto
    teste_resultado["previsao_referencia"] = y_referencia

    previsoes_teste.append(teste_resultado)

    # ======================================
    # 6. PREVISÃO PARA 2025
    # ======================================

    # Reajustar o modelo com todo o histórico disponível
    X_completo = X.to_numpy(dtype=float)
    X_completo = np.column_stack([
        np.ones(len(X_completo)),
        X_completo
    ])

    y_completo = dados["carga_media"].to_numpy()

    coeficientes_completos = np.linalg.lstsq(
        X_completo,
        y_completo,
        rcond=None
    )[0]

    # Criar os 12 meses de 2025
    futuro = pd.DataFrame({
        "data": pd.date_range("2025-01-01", periods=12, freq="MS")
    })

    futuro["tendencia"] = np.arange(
        len(dados), len(dados) + 12
    )

    futuro["mes"] = futuro["data"].dt.month

    meses_futuros = pd.get_dummies(
        futuro["mes"],
        prefix="mes",
        drop_first=True,
        dtype=int
    )

    meses_futuros = meses_futuros.reindex(
        columns=meses.columns,
        fill_value=0
    )

    X_futuro = pd.concat(
        [futuro[["tendencia"]], meses_futuros],
        axis=1
    )

    X_futuro = np.column_stack([
        np.ones(len(X_futuro)),
        X_futuro.to_numpy(dtype=float)
    ])

    futuro["subsistema"] = subsistema
    futuro["carga_prevista"] = (
        X_futuro @ coeficientes_completos
    )

    previsoes_2025.append(futuro)

# ==========================================
# 7. RESULTADOS
# ==========================================

df_resultados = pd.DataFrame(resultados)
df_teste = pd.concat(previsoes_teste, ignore_index=True)
df_2025 = pd.concat(previsoes_2025, ignore_index=True)

df_resultados.to_csv("metricas_modelo2.csv", index=False)
df_teste.to_csv("previsoes_teste_modelo2.csv", index=False)
df_2025.to_csv("previsao_2025_modelo2.csv", index=False)

print("\n========== MÉTRICAS DO MODELO ==========")
print(df_resultados.round(2).to_string(index=False))

print("\n========== AMOSTRA DAS PREVISÕES DE 2025 ==========")
print(df_2025.head(12).round(2).to_string(index=False))

print("\nArquivos gerados:")
print("- metricas_modelo2.csv")
print("- previsoes_teste_modelo2.csv")
print("- previsao_2025_modelo2.csv")
