#!/usr/bin/env python3
"""
Módulo de preparação e engenharia de atributos da camada Gold (db_energia_clima_puc).

- Extrai os dados integrados de carga diária (ONS) + clima diário agregado por subsistema (INMET/IBGE)
  diretamente do Amazon Athena e salva cache local em data/raw/gold_carga_clima_diario.csv.
- Constrói atributos de calendário, feriados nacionais e defasagens temporais (lags e médias móveis)
  respeitando rigorosamente a ordem cronológica por subsistema (zero vazamento de dados futuros).
"""

import datetime
import hashlib
import hmac
import io
import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = (
    BASE_DIR
    if (BASE_DIR / "data").exists() or not (BASE_DIR.parent / "data").exists()
    else BASE_DIR.parent
)
RAW_CSV_PATH = ROOT_DIR / "data" / "raw" / "gold_carga_clima_diario.csv"
PROCESSED_CSV_PATH = ROOT_DIR / "data" / "processed" / "base_modelagem_diaria.csv"

BUCKET = "projeto-puc-energia-clima-2026-689798927674"
DATABASE = "db_energia_clima_puc"
ATHENA_OUT = f"s3://{BUCKET}/athena-results/"

QUERY_GOLD_CARGA_CLIMA = """
WITH clima_subsistema AS (
    SELECT
        m.event_date,
        CASE
            WHEN loc.sigla_uf IN ('PR', 'RS', 'SC') THEN 'S'
            WHEN loc.sigla_uf IN ('SP', 'RJ', 'MG', 'ES', 'DF', 'GO', 'MS', 'MT', 'AC', 'RO') THEN 'SE'
            WHEN loc.sigla_uf IN ('AL', 'BA', 'CE', 'PB', 'PE', 'PI', 'RN', 'SE') THEN 'NE'
            WHEN loc.sigla_uf IN ('AM', 'AP', 'MA', 'PA', 'RR', 'TO') THEN 'N'
        END AS id_subsistema,
        AVG(CASE WHEN m.temperatura_maxima BETWEEN -10 AND 55 THEN m.temperatura_maxima END) AS temp_max,
        AVG(CASE WHEN m.temperatura_media  BETWEEN -10 AND 50 THEN m.temperatura_media  END) AS temp_media,
        AVG(CASE WHEN m.temperatura_minima BETWEEN -15 AND 45 THEN m.temperatura_minima END) AS temp_min,
        AVG(CASE WHEN m.precipitacao_total >= 0               THEN m.precipitacao_total END) AS precipitacao_total,
        AVG(CASE WHEN m.radiacao_global    BETWEEN 0 AND 4500 THEN m.radiacao_global    END) AS radiacao_global,
        AVG(CASE WHEN m.vento_velocidade_media >= 0           THEN m.vento_velocidade_media END) AS vento_vel_media
    FROM db_energia_clima_puc.gold_fato_clima m
    INNER JOIN db_energia_clima_puc.gold_dim_localizacao loc
        ON CAST(m.id_estacao AS VARCHAR) = loc.id_estacao
    WHERE m.event_date BETWEEN DATE '2016-01-01' AND DATE '2024-12-31'
    GROUP BY 1, 2
)
SELECT
    c.id_subsistema,
    c.nom_subsistema,
    c.event_date,
    c.val_cargaenergiamwmed,
    cl.temp_max,
    cl.temp_media,
    cl.temp_min,
    cl.precipitacao_total,
    cl.radiacao_global,
    cl.vento_vel_media
FROM db_energia_clima_puc.gold_fato_carga c
LEFT JOIN clima_subsistema cl
    ON  c.id_subsistema = cl.id_subsistema
    AND c.event_date    = cl.event_date
WHERE c.event_date BETWEEN DATE '2016-01-01' AND DATE '2024-12-31'
ORDER BY c.id_subsistema, c.event_date
"""


def _load_env() -> None:
    for candidate in (BASE_DIR / ".env", ROOT_DIR / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _aws_sigv4_request(
    service: str,
    method: str,
    host: str,
    uri: str,
    headers_extra: dict | None = None,
    payload: bytes = b"",
) -> bytes:
    _load_env()
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    session_token = os.environ.get("AWS_SESSION_TOKEN", "")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    t = datetime.datetime.now(datetime.timezone.utc)
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(payload).hexdigest()

    headers = {"host": host, "x-amz-date": amz_date}
    if session_token:
        headers["x-amz-security-token"] = session_token
    if service == "s3":
        headers["x-amz-content-sha256"] = payload_hash
    if headers_extra:
        for k, v in headers_extra.items():
            headers[k.lower()] = v

    sorted_keys = sorted(headers.keys())
    canonical_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in sorted_keys)
    signed_headers = ";".join(sorted_keys)
    canonical_request = "\n".join([method, uri, "", canonical_headers, signed_headers, payload_hash])

    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    req_headers = {k: v for k, v in headers.items() if k != "host"}
    req_headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    req = urllib.request.Request(
        f"https://{host}{uri}",
        data=payload if method == "POST" else None,
        headers=req_headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def extrair_do_athena(forcar_atualizacao: bool = False) -> Path:
    """Executa a query na camada Gold do Athena e baixa o CSV resultante do S3."""
    if RAW_CSV_PATH.exists() and RAW_CSV_PATH.stat().st_size > 0 and not forcar_atualizacao:
        return RAW_CSV_PATH

    _load_env()
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    athena_host = f"athena.{region}.amazonaws.com"

    print("Consultando camada Gold no Amazon Athena (db_energia_clima_puc)...")
    start_body = json.dumps({
        "ClientRequestToken": str(uuid.uuid4()),
        "QueryString": QUERY_GOLD_CARGA_CLIMA,
        "QueryExecutionContext": {"Database": DATABASE},
        "ResultConfiguration": {"OutputLocation": ATHENA_OUT},
    }).encode("utf-8")

    raw_start = _aws_sigv4_request(
        "athena",
        "POST",
        athena_host,
        "/",
        headers_extra={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AmazonAthena.StartQueryExecution",
        },
        payload=start_body,
    )
    qid = json.loads(raw_start.decode("utf-8"))["QueryExecutionId"]

    while True:
        raw_st = _aws_sigv4_request(
            "athena",
            "POST",
            athena_host,
            "/",
            headers_extra={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AmazonAthena.GetQueryExecution",
            },
            payload=json.dumps({"QueryExecutionId": qid}).encode("utf-8"),
        )
        qexec = json.loads(raw_st.decode("utf-8"))["QueryExecution"]
        state = qexec["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)

    if state != "SUCCEEDED":
        reason = qexec["Status"].get("StateChangeReason", "sem detalhes")
        raise RuntimeError(f"Falha na query do Athena ({state}): {reason}")

    output_s3 = qexec["ResultConfiguration"]["OutputLocation"]
    # output_s3 formato: s3://<bucket>/athena-results/<qid>.csv
    key = output_s3.replace(f"s3://{BUCKET}/", "", 1)
    s3_host = f"{BUCKET}.s3.{region}.amazonaws.com"
    csv_bytes = _aws_sigv4_request("s3", "GET", s3_host, "/" + urllib.parse.quote(key, safe="/-_.~"))

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_CSV_PATH.write_bytes(csv_bytes)
    print(f"Base extraída e salva em: {RAW_CSV_PATH} ({len(csv_bytes) / 1024:.1f} KB)")
    return RAW_CSV_PATH


def montar_base(forcar_athena: bool = False) -> pd.DataFrame:
    """Carrega a base diária integrada e adiciona variáveis de calendário e tendência."""
    csv_path = extrair_do_athena(forcar_atualizacao=forcar_athena)
    df = pd.read_csv(csv_path, parse_dates=["event_date"])
    df = df.sort_values(["id_subsistema", "event_date"]).reset_index(drop=True)

    feriados_br = holidays.Brazil(
        years=range(2016, 2026),
        categories=(holidays.PUBLIC, holidays.OPTIONAL),
    )
    datas_feriado = pd.to_datetime(list(feriados_br.keys()))

    df["ano"] = df["event_date"].dt.year
    df["mes"] = df["event_date"].dt.month
    df["dia_semana"] = df["event_date"].dt.day_of_week
    df["dia_ano"] = df["event_date"].dt.day_of_year
    df["feriado"] = df["event_date"].isin(datas_feriado).astype(int)
    df["tendencia"] = (df["event_date"] - pd.Timestamp("2016-01-01")).dt.days
    return df


def criar_defasagens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria defasagens temporais (lags e médias móveis) estritamente causais por subsistema,
    sem nenhum vazamento de informação futura.
    """
    df = df.sort_values(["id_subsistema", "event_date"]).copy()
    g_carga = df.groupby("id_subsistema")["val_cargaenergiamwmed"]
    g_temp = df.groupby("id_subsistema")["temp_media"]

    df["carga_lag_1"] = g_carga.shift(1)
    df["carga_lag_7"] = g_carga.shift(7)
    df["carga_media_7d"] = g_carga.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())

    df["temp_media_lag_1"] = g_temp.shift(1)
    df["temp_media_3d"] = g_temp.transform(lambda s: s.rolling(3, min_periods=1).mean())

    # Remove apenas os 7 primeiros dias de jan/2016 onde carga_lag_7 / carga_media_7d ainda não existem
    df = df.dropna(subset=["carga_lag_1", "carga_lag_7", "carga_media_7d"]).reset_index(drop=True)

    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    return df


def separar(df: pd.DataFrame, modo: str = "2024") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separação estritamente cronológica (sem embaralhamento):
      - modo='2024' (padrão dos scripts do orientador e Etapa 1):
          Treino inicial = 2016..2022 | Validação de hiperparâmetros = 2023 |
          (Treino completo = 2016..2023) | Teste final = 2024
      - modo='2023_2024' (janela estendida de 2 anos do modelo2.py):
          Treino inicial = 2016..2021 | Validação de hiperparâmetros = 2022 |
          (Treino completo = 2016..2022) | Teste final = 2023..2024
    """
    anos = df["event_date"].dt.year
    if modo == "2023_2024":
        treino = df[anos <= 2021].copy()
        validacao = df[anos == 2022].copy()
        teste = df[anos.between(2023, 2024)].copy()
    elif modo == "2024":
        treino = df[anos <= 2022].copy()
        validacao = df[anos == 2023].copy()
        teste = df[anos == 2024].copy()
    else:
        raise ValueError(f"Modo inválido: {modo}")
    return treino, validacao, teste


if __name__ == "__main__":
    base = criar_defasagens(montar_base())
    print(f"Base processada pronta: {len(base)} registros | Colunas: {list(base.columns)}")
    for m in ("2023_2024", "2024"):
        tr, va, te = separar(base, modo=m)
        print(f"  [{m}] treino={len(tr)} | validacao={len(va)} | teste={len(te)}")
