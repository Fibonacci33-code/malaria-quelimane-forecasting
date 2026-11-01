import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PROTOTIPO EXPERIMENTAL
# PREVISAO MENSAL DA MALARIA — QUELIMANE, MOCAMBIQUE
# ============================================================

st.set_page_config(
    page_title="Malaria Forecasting — Quelimane",
    page_icon="📊",
    layout="wide"
)

st.title("Malaria Forecasting and Climate Monitoring")
st.subheader("Quelimane City, Mozambique")

st.warning(
    "PUBLIC DEMONSTRATION ? SYNTHETIC DATA ONLY. "
    "All epidemiological and climate values displayed in this "
    "demonstration are artificially generated. "
    "They do not represent observed malaria data from Quelimane. "
    "Forecasts and model outputs in this demonstration must not "
    "be interpreted as scientific findings or used for "
    "public health decisions."
)


st.info(
    "Experimental research prototype. "
    "Predictions represent reported malaria-positive "
    "test results, not population-based incidence rates. "
    "This system is not prospectively validated."
)


# ============================================================
# 1. LOCALIZACAO DOS FICHEIROS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
base = BASE_DIR / "data" / "malaria"

entrada = (
    base / "Quelimane_clima_malaria_integrado_2019_2025(1).xlsx"
)

ficheiro_metricas = (
    base / "comparacao_todos_modelos_2024_2025.csv"
)

ficheiro_previsoes = (
    base / "previsoes_random_forest_2024_2025.csv"
)

# Registo das previsoes experimentais
pasta_historico = base / "forecast_history"
ficheiro_historico = pasta_historico / "forecast_registry.csv"

COLUNAS_HISTORICO = [
    "forecast_id",
    "generated_at_utc",
    "forecast_month",
    "last_observed_month",
    "predicted_positive_results",
    "model",
    "data_source",
    "evaluation_status",
    "forecast_classification",
    "dataset_fingerprint",
]


# Registo controlado das actualizacoes mensais (independente das previsoes)
pasta_actualizacoes = base / "update_history"
ficheiro_actualizacoes = pasta_actualizacoes / "update_registry.csv"

COLUNAS_ACTUALIZACOES = [
    "update_id",
    "registered_at_utc",
    "filename",
    "data_source",
    "verification_status",
    "previous_last_month",
    "new_last_month",
    "months_added",
    "dataset_fingerprint",
]


def carregar_actualizacoes():
    """Consulta o registo sem alterar as actualizacoes existentes."""
    if not ficheiro_actualizacoes.exists() or ficheiro_actualizacoes.stat().st_size == 0:
        return pd.DataFrame(columns=COLUNAS_ACTUALIZACOES)
    registos = pd.read_csv(ficheiro_actualizacoes, dtype=str, keep_default_na=False)
    faltantes = [c for c in COLUNAS_ACTUALIZACOES if c not in registos.columns]
    if faltantes:
        raise ValueError(f"Colunas em falta no historico de actualizacoes: {faltantes}")
    return registos[COLUNAS_ACTUALIZACOES]


def guardar_actualizacao(registo):
    """Grava so apos confirmacao; impede repetir a mesma base validada."""
    existentes = carregar_actualizacoes()
    if not existentes.empty and (
        existentes["dataset_fingerprint"].astype(str) == registo["dataset_fingerprint"]
    ).any():
        return False
    pasta_actualizacoes.mkdir(parents=True, exist_ok=True)
    actualizado = pd.concat(
        [existentes, pd.DataFrame([registo])], ignore_index=True
    )[COLUNAS_ACTUALIZACOES]
    temporario = ficheiro_actualizacoes.with_suffix(".tmp")
    actualizado.to_csv(temporario, index=False, encoding="utf-8")
    temporario.replace(ficheiro_actualizacoes)
    return True


# Avaliacao de previsoes arquivadas (ficheiro separado dos historicos)
pasta_avaliacoes = base / "prospective_evaluation"
ficheiro_avaliacoes = pasta_avaliacoes / "evaluation_registry.csv"

COLUNAS_AVALIACOES = [
    "evaluation_id", "forecast_id", "forecast_month", "generated_at_utc",
    "evaluated_at_utc", "predicted_positive_results",
    "observed_positive_results", "absolute_error", "percentage_error",
    "evaluation_type", "data_source", "verification_status",
]


def carregar_avaliacoes():
    """Consulta o registo de avaliacoes sem alterar ficheiros existentes."""
    if not ficheiro_avaliacoes.exists() or ficheiro_avaliacoes.stat().st_size == 0:
        return pd.DataFrame(columns=COLUNAS_AVALIACOES)
    avaliacoes = pd.read_csv(
        ficheiro_avaliacoes, dtype=str, keep_default_na=False,
        encoding="utf-8-sig",
    )
    ausentes = [c for c in COLUNAS_AVALIACOES if c not in avaliacoes.columns]
    if ausentes:
        raise ValueError(f"Colunas em falta no registo de avaliacoes: {ausentes}")
    return avaliacoes[COLUNAS_AVALIACOES]


def guardar_avaliacao(registo):
    """Registo explicito e unico por previsao; escrita via ficheiro temporario."""
    anteriores = carregar_avaliacoes()
    if not anteriores.empty and (
        anteriores["forecast_id"].astype(str) == str(registo["forecast_id"])
    ).any():
        return False
    pasta_avaliacoes.mkdir(parents=True, exist_ok=True)
    completo = pd.concat(
        [anteriores, pd.DataFrame([registo])], ignore_index=True
    )[COLUNAS_AVALIACOES]
    temporario = ficheiro_avaliacoes.with_suffix(".tmp")
    completo.to_csv(temporario, index=False, encoding="utf-8")
    temporario.replace(ficheiro_avaliacoes)
    return True


# ============================================================
# 2. CARREGAR E VALIDAR DADOS
# ============================================================

@st.cache_data
def carregar_dados(caminho):
    return pd.read_excel(caminho)


def calcular_fingerprint_dataset(dados):
    """Gera uma assinatura para identificar a base usada na previsao."""
    conteudo = pd.util.hash_pandas_object(
        dados.astype(str),
        index=True,
    ).values.tobytes()
    return hashlib.sha256(conteudo).hexdigest()


def carregar_historico():
    """Carrega o histórico e acomoda o cabeçalho inicial do CSV."""
    pasta_historico.mkdir(parents=True, exist_ok=True)

    if not ficheiro_historico.exists() or ficheiro_historico.stat().st_size == 0:
        return pd.DataFrame(columns=COLUNAS_HISTORICO)

    historico = pd.read_csv(ficheiro_historico, dtype=str)

    for coluna in COLUNAS_HISTORICO:
        if coluna not in historico.columns:
            historico[coluna] = ""

    return historico[COLUNAS_HISTORICO]


def guardar_previsao(historico, registo):
    """Guarda uma previsão e impede duplicação para a mesma base."""
    chave = [
        "forecast_month",
        "last_observed_month",
        "model",
        "dataset_fingerprint",
    ]

    if not historico.empty:
        duplicado = pd.Series(True, index=historico.index)
        for coluna in chave:
            duplicado &= (
                historico[coluna].fillna("").astype(str)
                == str(registo[coluna])
            )
        if duplicado.any():
            return False

    actualizado = pd.concat(
        [historico, pd.DataFrame([registo])],
        ignore_index=True,
    )[COLUNAS_HISTORICO]

    pasta_historico.mkdir(parents=True, exist_ok=True)
    temporario = ficheiro_historico.with_suffix(".tmp")
    actualizado.to_csv(temporario, index=False, encoding="utf-8")
    temporario.replace(ficheiro_historico)
    return True


def validar_actualizacao(dados_novos, dados_originais):
    """Valida importacao mensal, preservando integralmente o historico."""
    epidemiologicas = [
        "HTZ positive",
        "TDR positive",
        "Confirmed (HTZ + TDR)",
        "Rapid tests used",
    ]
    climaticas = [
        "Precipitation (mm)",
        "Mean temperature (°C)",
        "Minimum temperature (°C)",
        "Maximum temperature (°C)",
        "Relative humidity (%)",
    ]
    obrigatorias = ["Year", "Month"] + epidemiologicas + climaticas

    faltantes = [c for c in obrigatorias if c not in dados_novos.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias em falta: {faltantes}")

    # Impedir eliminacao de colunas historicas (inclusive metadados).
    colunas_originais = list(dados_originais.columns)
    removidas = [c for c in colunas_originais if c not in dados_novos.columns]
    if removidas:
        raise ValueError(f"Colunas históricas em falta: {removidas}")

    def preparar(dados):
        dados = dados.copy()
        for c in ["Year", "Month"]:
            dados[c] = pd.to_numeric(dados[c], errors="raise")
            if dados[c].isna().any() or not np.isfinite(dados[c]).all():
                raise ValueError(f"Valores inválidos em {c}.")
            if (dados[c] % 1 != 0).any():
                raise ValueError(f"{c} deve conter números inteiros.")
        if (~dados["Month"].between(1, 12)).any():
            raise ValueError("Meses fora do intervalo de 1 a 12.")
        dados["date"] = pd.to_datetime(
            dict(year=dados["Year"].astype(int),
                 month=dados["Month"].astype(int), day=1),
            errors="raise",
        )
        if dados["date"].duplicated().any():
            raise ValueError("Existem meses duplicados.")
        dados = dados.sort_values("date").reset_index(drop=True)
        for c in epidemiologicas + climaticas:
            dados[c] = pd.to_numeric(dados[c], errors="raise")
        if dados[epidemiologicas + climaticas].isna().any().any():
            raise ValueError("Existem valores ausentes.")
        if not np.isfinite(dados[epidemiologicas + climaticas].to_numpy(dtype=float)).all():
            raise ValueError("Existem valores numéricos inválidos.")
        colunas_nao_negativas = epidemiologicas + [
            "Precipitation (mm)",
            "Relative humidity (%)",
        ]

        if (dados[colunas_nao_negativas] < 0).any().any():
            raise ValueError(
                "Existem valores negativos em contagens, "
                "precipitacao ou humidade relativa."
            )
        if (dados[epidemiologicas].to_numpy(dtype=float) % 1 != 0).any():
            raise ValueError("As contagens epidemiológicas devem ser inteiras.")
        if not (dados["HTZ positive"] + dados["TDR positive"] ==
                dados["Confirmed (HTZ + TDR)"]).all():
            raise ValueError(
                "Inconsistência epidemiológica: HTZ positive + TDR positive "
                "deve ser igual a Confirmed (HTZ + TDR)."
            )
        if (dados["TDR positive"] > dados["Rapid tests used"]).any():
            raise ValueError(
                "Inconsistência epidemiológica: TDR positive não pode "
                "ultrapassar Rapid tests used."
            )
        if (~dados["Relative humidity (%)"].between(0, 100)).any():
            raise ValueError("Humidade relativa fora de 0 a 100%.")
        temperatura_minima = dados["Minimum temperature (°C)"]
        temperatura_media = dados["Mean temperature (°C)"]
        temperatura_maxima = dados["Maximum temperature (°C)"]

        if (
            (temperatura_minima > temperatura_media)
            | (temperatura_media > temperatura_maxima)
        ).any():
            raise ValueError(
                "Inconsistencia climatica: a temperatura minima "
                "deve ser menor ou igual a temperatura media, "
                "e a temperatura media deve ser menor ou igual "
                "a temperatura maxima."
            )
        return dados

    novo = preparar(dados_novos)
    original = preparar(dados_originais)
    if len(novo) < len(original):
        raise ValueError("O ficheiro actualizado contém menos meses do que a base original.")

    historico = novo.iloc[:len(original)].reset_index(drop=True)
    if not historico["date"].equals(original["date"]):
        raise ValueError("Os meses históricos foram alterados.")

    # Comparar todas as colunas originais, nao apenas as usadas no modelo.
    for c in colunas_originais:
        if c in ["Year", "Month"] + epidemiologicas + climaticas:
            iguais = np.array_equal(
                historico[c].to_numpy(), original[c].to_numpy(),
                equal_nan=True,
            )
        else:
            iguais = historico[c].equals(original[c])
        if not iguais:
            raise ValueError(f"Os valores históricos da coluna '{c}' foram alterados.")

    esperados = pd.date_range(original["date"].min(), novo["date"].max(), freq="MS")
    if not pd.DatetimeIndex(novo["date"]).equals(esperados):
        raise ValueError("A série mensal contém lacunas.")

    return novo.drop(columns=["date"])


if not entrada.exists():
    st.error(
        "Integrated malaria-climate dataset not found."
    )
    st.stop()

# ============================================================
# IMPORTACAO DE DADOS ACTUALIZADOS
# ============================================================

dados_originais = carregar_dados(entrada)

st.sidebar.header("Data Management")

ficheiro_actualizado = st.sidebar.file_uploader(
    "Upload Updated Excel Dataset",
    type=["xlsx"],
    help=(
        "Upload an updated dataset containing the "
        "original historical observations and any "
        "additional consecutive months."
    )
)

df = dados_originais.copy()

if ficheiro_actualizado is not None:

    try:
        dados_importados = pd.read_excel(
            ficheiro_actualizado
        )

        df = validar_actualizacao(
            dados_importados,
            dados_originais
        )

        meses_adicionados = (
            len(df) - len(dados_originais)
        )

        if meses_adicionados == 0:
            st.sidebar.info(
                "Dataset validated. No new months were added."
            )
        else:
            st.sidebar.success(
                f"Dataset validated: "
                f"{meses_adicionados} new month(s) added."
            )

            # A validacao e obrigatoria; o registo e sempre manual.
            st.sidebar.caption(
                "Dataset provenance is self-declared and remains unverified "
                "until checked against official source documents."
            )
            origem_declarada = st.sidebar.text_input(
                "Declared data source (institution/document)",
                key="update_declared_source",
                placeholder="e.g., district health surveillance report",
            )
            st.sidebar.caption(
                "Registration does not replace or modify the original Excel dataset."
            )
            if st.sidebar.button(
                "Register Dataset Update",
                key="register_dataset_update",
            ):
                if not origem_declarada.strip():
                    st.sidebar.warning(
                        "Enter the declared data source before registering."
                    )
                else:
                    try:
                        anterior = pd.to_datetime(
                            dict(
                                year=dados_originais["Year"].astype(int),
                                month=dados_originais["Month"].astype(int),
                                day=1,
                            )
                        ).max()
                        actual = pd.to_datetime(
                            dict(
                                year=df["Year"].astype(int),
                                month=df["Month"].astype(int),
                                day=1,
                            )
                        ).max()
                        registo_actualizacao = {
                            "update_id": str(uuid.uuid4()),
                            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                            "filename": ficheiro_actualizado.name,
                            "data_source": origem_declarada.strip(),
                            "verification_status": "unverified",
                            "previous_last_month": anterior.strftime("%Y-%m"),
                            "new_last_month": actual.strftime("%Y-%m"),
                            "months_added": str(meses_adicionados),
                            "dataset_fingerprint": calcular_fingerprint_dataset(
                                df.sort_values(["Year", "Month"]).reset_index(drop=True)
                            ),
                        }
                        if guardar_actualizacao(registo_actualizacao):
                            st.sidebar.success("Dataset update registered successfully.")
                        else:
                            st.sidebar.info(
                                "This dataset is already registered. "
                                "No duplicate record was created."
                            )
                    except Exception as erro_registo:
                        st.sidebar.error(
                            f"Could not register dataset update: {erro_registo}"
                        )

    except Exception as erro:

        st.error(
            f"Dataset validation failed: {erro}"
        )

        st.warning(
            "The uploaded dataset was rejected. "
            "The original research dataset remains unchanged."
        )

        st.stop()

# Historico visivel sem necessidade de carregar um ficheiro.
st.sidebar.subheader("Dataset Update History")
try:
    historico_actualizacoes = carregar_actualizacoes()
    if historico_actualizacoes.empty:
        st.sidebar.info("No dataset updates have been registered yet.")
    else:
        st.sidebar.dataframe(
            historico_actualizacoes.drop(columns=["dataset_fingerprint"]),
            hide_index=True,
            use_container_width=True,
        )
except Exception as erro_historico:
    st.sidebar.error(f"Could not read dataset update history: {erro_historico}")

clima_cols = [
    "Precipitation (mm)",
    "Mean temperature (°C)",
    "Relative humidity (%)"
]

colunas_necessarias = [
    "Year",
    "Month",
    "Confirmed (HTZ + TDR)"
] + clima_cols

faltantes = [
    coluna
    for coluna in colunas_necessarias
    if coluna not in df.columns
]

if faltantes:
    st.error(
        f"Missing columns in the dataset: {faltantes}"
    )
    st.stop()

df["date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=1
    )
)

df = df.sort_values("date").set_index("date")

if df.index.has_duplicates:
    st.error(
        "Duplicate monthly observations detected."
    )
    st.stop()

df = df.asfreq("MS")

colunas_numericas = [
    "Confirmed (HTZ + TDR)"
] + clima_cols

for coluna in colunas_numericas:
    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )

if df[colunas_numericas].isna().any().any():
    st.error(
        "Missing or invalid values detected in the "
        "malaria-climate dataset."
    )
    st.stop()

y = df["Confirmed (HTZ + TDR)"].astype(float)


# ============================================================
# 3. PAINEL EPIDEMIOLOGICO
# ============================================================

st.header("1. Malaria Surveillance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Months available",
    len(df)
)

col2.metric(
    "Total positive results",
    f"{int(y.sum()):,}"
)

col3.metric(
    "Latest monthly positive results",
    f"{int(y.iloc[-1]):,}"
)

st.subheader("Historical Malaria-Positive Results")

st.line_chart(
    y.rename("Reported malaria-positive results")
)


# ============================================================
# 4. PAINEL CLIMATICO
# ============================================================

st.header("2. Climate Monitoring")

variavel = st.selectbox(
    "Select climate variable",
    clima_cols
)

st.line_chart(
    df[variavel]
)


# ============================================================
# 5. CONSTRUIR PREDITORES
# ============================================================

features = pd.DataFrame(index=df.index)

features["month_sin"] = np.sin(
    2 * np.pi * df.index.month / 12
)

features["month_cos"] = np.cos(
    2 * np.pi * df.index.month / 12
)

for lag in [1, 2, 3, 12]:
    features[f"malaria_lag_{lag}"] = y.shift(lag)

for coluna in clima_cols:
    for lag in [1, 2, 3]:
        features[f"{coluna}_lag_{lag}"] = (
            df[coluna].shift(lag)
        )

dataset = features.copy()
dataset["target"] = y
dataset = dataset.dropna()

X = dataset.drop(columns=["target"])
target = dataset["target"]


# ============================================================
# 6. TREINAR RANDOM FOREST
# ============================================================

@st.cache_resource
def treinar_modelo(X_train, y_train):

    modelo = RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    return modelo


modelo = treinar_modelo(X, target)


# ============================================================
# 7. PREPARAR PREVISAO DO MES SEGUINTE
# ============================================================

ultimo_mes = df.index.max()

proximo_mes = (
    ultimo_mes + pd.DateOffset(months=1)
)

nova_linha = {}

nova_linha["month_sin"] = np.sin(
    2 * np.pi * proximo_mes.month / 12
)

nova_linha["month_cos"] = np.cos(
    2 * np.pi * proximo_mes.month / 12
)

for lag in [1, 2, 3, 12]:

    data_historica = (
        proximo_mes - pd.DateOffset(months=lag)
    )

    nova_linha[f"malaria_lag_{lag}"] = float(
        y.loc[data_historica]
    )

for coluna in clima_cols:

    for lag in [1, 2, 3]:

        data_historica = (
            proximo_mes - pd.DateOffset(months=lag)
        )

        nova_linha[f"{coluna}_lag_{lag}"] = float(
            df.loc[data_historica, coluna]
        )

X_futuro = pd.DataFrame(
    [nova_linha],
    index=[proximo_mes]
)

X_futuro = X_futuro[X.columns]

previsao = float(
    modelo.predict(X_futuro)[0]
)

previsao = max(0.0, previsao)


# ============================================================
# 8. APRESENTAR PREVISAO
# ============================================================

st.header("3. One-Month-Ahead Forecast")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Last observed month",
    ultimo_mes.strftime("%B %Y")
)

col2.metric(
    "Forecast month",
    proximo_mes.strftime("%B %Y")
)

col3.metric(
    "Predicted malaria-positive results",
    f"{previsao:,.0f}"
)

st.caption(
    "Retrospective experimental forecast generated "
    "using lagged malaria surveillance and climate "
    "observations. Future climate values are not required."
)

st.warning(
    "This is an experimental one-month-ahead forecast based on "
    "the latest available dataset. It has not been prospectively "
    "validated and must not be treated as an operational alert."
)

# ------------------------------------------------------------
# REGISTO MANUAL E CONTROLADO DA PREVISAO
# ------------------------------------------------------------

st.subheader("Forecast History")

st.caption(
    "Save a forecast explicitly. Re-running or refreshing the app "
    "does not create a history record. Saved forecasts are retained "
    "for later comparison with observed data."
)

origem_dados = (
    "uploaded_excel_unverified"
    if ficheiro_actualizado is not None
    else "original_dataset_unverified"
)

fingerprint_dataset = calcular_fingerprint_dataset(df)
historico_previsoes = carregar_historico()

if st.button(
    "Save Forecast to History",
    key="save_forecast_to_history",
    type="primary",
):
    registo = {
        "forecast_id": str(uuid.uuid4()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_month": proximo_mes.strftime("%Y-%m"),
        "last_observed_month": ultimo_mes.strftime("%Y-%m"),
        "predicted_positive_results": f"{previsao:.6f}",
        "model": "RandomForestRegressor",
        "data_source": origem_dados,
        "evaluation_status": "awaiting_observed_data",
        "forecast_classification": "experimental_unverified",
        "dataset_fingerprint": fingerprint_dataset,
    }

    try:
        guardada = guardar_previsao(historico_previsoes, registo)
        if guardada:
            st.success(
                "Forecast saved to history. This experimental record "
                "awaits observed data for evaluation."
            )
            historico_previsoes = carregar_historico()
        else:
            st.info(
                "This forecast for the same month, training endpoint, "
                "model and dataset has already been saved. No duplicate "
                "record was created."
            )
    except Exception as erro:
        st.error(f"Could not save forecast history: {erro}")

historico_previsoes = carregar_historico()

if not historico_previsoes.empty:
    st.dataframe(
        historico_previsoes.drop(columns=["dataset_fingerprint"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No forecasts have been saved yet.")


# ============================================================
# AVALIACAO CONTROLADA DAS PREVISOES ARQUIVADAS — PONTO 3
# ============================================================

st.subheader("Forecast Evaluation")
st.caption(
    "Evaluation is separate from model training and historical validation. "
    "A saved forecast is compared only with an observed value for its "
    "exact target month. Dataset consistency checks do not verify provenance."
)

try:
    avaliacoes_guardadas = carregar_avaliacoes()
    if not avaliacoes_guardadas.empty:
        st.dataframe(
            avaliacoes_guardadas,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No forecast evaluations have been registered yet.")

    if historico_previsoes.empty:
        st.info("Save a forecast before preparing an evaluation.")
    else:
        previsoes_disponiveis = historico_previsoes.copy()
        previsoes_disponiveis["label"] = previsoes_disponiveis.apply(
            lambda r: (
                f'{r["forecast_month"]} | {r["forecast_id"]} '
                f'| saved {r["generated_at_utc"]}'
            ),
            axis=1,
        )
        escolha = st.selectbox(
            "Select archived forecast to evaluate",
            options=list(previsoes_disponiveis.index),
            format_func=lambda i: previsoes_disponiveis.loc[i, "label"],
            key="forecast_evaluation_choice",
        )
        seleccionada = previsoes_disponiveis.loc[escolha]
        mes_alvo = str(seleccionada["forecast_month"])
        id_previsao = str(seleccionada["forecast_id"])
        ja_avaliada = (
            not avaliacoes_guardadas.empty
            and (avaliacoes_guardadas["forecast_id"] == id_previsao).any()
        )
        if ja_avaliada:
            st.info("This forecast has already been evaluated; no duplicate is permitted.")
        else:
            # df foi validado previamente pelo fluxo de importacao.
            data_alvo = pd.to_datetime(mes_alvo + "-01", errors="raise")
            if data_alvo not in df.index:
                st.warning(
                    f"No observed results for {mes_alvo} in the active dataset. "
                    "Evaluation remains pending."
                )
            else:
                observado = float(df.loc[data_alvo, "Confirmed (HTZ + TDR)"])
                previsto = float(seleccionada["predicted_positive_results"])
                if not (np.isfinite(observado) and np.isfinite(previsto)):
                    raise ValueError("Invalid forecast or observed value.")
                erro_absoluto = abs(observado - previsto)
                erro_percentual = (
                    100 * erro_absoluto / observado if observado > 0 else None
                )
                gerado_em = pd.to_datetime(
                    seleccionada["generated_at_utc"], utc=True, errors="raise"
                )
                # Uma previsao registada apos o inicio do mes-alvo nao
                # pode ser classificada como prospectiva.
                tipo_avaliacao = (
                    "prospective_candidate"
                    if gerado_em < pd.Timestamp(data_alvo, tz="UTC")
                    else "retrospective_experimental"
                )
                # O Excel original e uma base de investigacao nao verificada;
                # uploads exigem registo e verificacao documental independente.
                if ficheiro_actualizado is None:
                    origem_observada = "original_research_dataset"
                    verificado = False
                else:
                    assinatura = calcular_fingerprint_dataset(
                        df.reset_index().drop(columns=["date"]).sort_values(
                            ["Year", "Month"]
                        ).reset_index(drop=True)
                    )
                    correspondentes = historico_actualizacoes.loc[
                        historico_actualizacoes["dataset_fingerprint"] == assinatura
                    ] if "historico_actualizacoes" in globals() else pd.DataFrame()
                    verificado = (
                        not correspondentes.empty
                        and correspondentes["verification_status"].str.lower().eq(
                            "verified"
                        ).any()
                    )
                    origem_observada = (
                        str(correspondentes.iloc[-1]["data_source"])
                        if not correspondentes.empty else "unregistered_uploaded_excel"
                    )

                st.write(
                    f"**Target:** {mes_alvo} · **Forecast:** {previsto:,.2f} · "
                    f"**Observed (active dataset):** {observado:,.0f}"
                )
                st.write(
                    f"**Absolute error:** {erro_absoluto:,.2f} · "
                    f"**Absolute percentage error:** "
                    f"{erro_percentual:.2f}%" if erro_percentual is not None
                    else (
                        f"**Absolute error:** {erro_absoluto:,.2f} · "
                        "**Absolute percentage error:** undefined (observed = 0)"
                    )
                )
                st.caption(f"Timing classification: {tipo_avaliacao}")
                if not verificado:
                    st.warning(
                        "PREVIEW ONLY — observed data are unverified or not "
                        "linked to an independently verified source. "
                        "These numbers are not scientific validation results. "
                        "Registration is disabled."
                    )
                else:
                    st.info(
                        "Source registry indicates verified status. Confirm the "
                        "underlying documentary evidence independently before saving."
                    )
                    if st.button(
                        "Save Verified Forecast Evaluation",
                        key="save_verified_forecast_evaluation",
                    ):
                        novo_registo = {
                            "evaluation_id": str(uuid.uuid4()),
                            "forecast_id": id_previsao,
                            "forecast_month": mes_alvo,
                            "generated_at_utc": str(seleccionada["generated_at_utc"]),
                            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "predicted_positive_results": f"{previsto:.6f}",
                            "observed_positive_results": f"{observado:.0f}",
                            "absolute_error": f"{erro_absoluto:.6f}",
                            "percentage_error": (
                                f"{erro_percentual:.6f}"
                                if erro_percentual is not None else ""
                            ),
                            "evaluation_type": tipo_avaliacao,
                            "data_source": origem_observada,
                            "verification_status": "verified_source_registry",
                        }
                        if guardar_avaliacao(novo_registo):
                            st.success("Verified evaluation saved.")
                        else:
                            st.info("Evaluation already exists; no duplicate saved.")
except Exception as erro_avaliacao:
    st.error(f"Forecast evaluation could not be prepared: {erro_avaliacao}")


# ============================================================
# 9. COMPARACAO DOS MODELOS
# ============================================================

st.header("4. Model Performance")

if ficheiro_metricas.exists():

    metricas = pd.read_csv(
        ficheiro_metricas
    )

    st.dataframe(
        metricas,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Historical rolling-origin validation: "
        "January 2024–December 2025; "
        "one-month-ahead forecasts."
    )

else:

    st.warning(
        "Model comparison results are not available."
    )


# ============================================================
# 10. OBSERVADO VS PREVISTO
# ============================================================

st.header("5. Observed vs Predicted Malaria Results")

st.write(
    "Random Forest rolling-origin validation "
    "(January 2024–December 2025)."
)

if ficheiro_previsoes.exists():

    previsoes = pd.read_csv(
        ficheiro_previsoes
    )

    # --------------------------------------------------------
    # IDENTIFICAR COLUNAS DO FICHEIRO
    # --------------------------------------------------------

    colunas = list(previsoes.columns)

    st.caption(
        "Columns detected in the historical "
        "prediction file: "
        + ", ".join(colunas)
    )

    # Nomes alternativos que podem existir
    alternativas_data = [
        "Month",
        "Date",
        "date",
        "month",
        "Year-month"
    ]

    alternativas_observado = [
        "Observed",
        "Actual",
        "Real",
        "y_true"
    ]

    alternativas_previsto = [
        "Predicted",
        "Prediction",
        "RF_Predicted",
        "Random_Forest",
        "RF_Com_Clima",
        "y_pred"
    ]

    def encontrar_coluna(alternativas, disponiveis):

        for nome in alternativas:
            if nome in disponiveis:
                return nome

        return None

    coluna_data = encontrar_coluna(
        alternativas_data,
        colunas
    )

    coluna_observado = encontrar_coluna(
        alternativas_observado,
        colunas
    )

    coluna_previsto = encontrar_coluna(
        alternativas_previsto,
        colunas
    )

    if (
        coluna_data is None
        or coluna_observado is None
        or coluna_previsto is None
    ):

        st.warning(
            "The historical prediction file was found, "
            "but its column names require verification."
        )

        st.dataframe(
            previsoes,
            use_container_width=True,
            hide_index=True
        )

    else:

        # ----------------------------------------------------
        # ORGANIZAR DADOS
        # ----------------------------------------------------

        comparacao = previsoes[
            [
                coluna_data,
                coluna_observado,
                coluna_previsto
            ]
        ].copy()

        comparacao.columns = [
            "Month",
            "Observed",
            "Predicted"
        ]

        comparacao["Month"] = pd.to_datetime(
            comparacao["Month"]
        )

        comparacao["Observed"] = pd.to_numeric(
            comparacao["Observed"],
            errors="coerce"
        )

        comparacao["Predicted"] = pd.to_numeric(
            comparacao["Predicted"],
            errors="coerce"
        )

        comparacao = comparacao.dropna()

        comparacao = comparacao.sort_values(
            "Month"
        )

        # ----------------------------------------------------
        # VALIDAR PERIODO HISTORICO
        # ----------------------------------------------------

        periodo_esperado = pd.date_range(
            "2024-01-01",
            "2025-12-01",
            freq="MS"
        )

        meses_observados = pd.DatetimeIndex(
            comparacao["Month"]
        )

        periodo_valido = (
            len(comparacao) == 24
            and meses_observados.equals(
                periodo_esperado
            )
        )

        if not periodo_valido:

            st.warning(
                "The historical predictions do not "
                "contain exactly the expected 24 "
                "monthly observations from January "
                "2024 to December 2025. "
                "Performance indicators were not calculated."
            )

            st.dataframe(
                comparacao,
                use_container_width=True,
                hide_index=True
            )

        else:

            # ------------------------------------------------
            # CALCULAR ERROS
            # ------------------------------------------------

            comparacao["Error"] = (
                comparacao["Observed"]
                - comparacao["Predicted"]
            )

            comparacao["Absolute_Error"] = (
                comparacao["Error"].abs()
            )

            mae = float(
                comparacao["Absolute_Error"].mean()
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        comparacao["Error"] ** 2
                    )
                )
            )

            # ------------------------------------------------
            # INDICADORES
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Validation months",
                len(comparacao)
            )

            c2.metric(
                "Random Forest MAE",
                f"{mae:,.2f}"
            )

            c3.metric(
                "Random Forest RMSE",
                f"{rmse:,.2f}"
            )

            # ------------------------------------------------
            # GRAFICO
            # ------------------------------------------------

            st.subheader(
                "Observed and Predicted Monthly Results"
            )

            grafico = comparacao.set_index(
                "Month"
            )[
                [
                    "Observed",
                    "Predicted"
                ]
            ]

            st.line_chart(
                grafico
            )

            # ------------------------------------------------
            # TABELA
            # ------------------------------------------------

            st.subheader(
                "Monthly Prediction Errors"
            )

            tabela = comparacao.copy()

            tabela["Month"] = (
                tabela["Month"].dt.strftime("%Y-%m")
            )

            st.dataframe(
                tabela,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Historical out-of-sample predictions "
                "generated during rolling-origin validation. "
                "The prototype does not recalculate these "
                "historical forecasts."
            )

else:

    st.warning(
        "Historical Random Forest predictions "
        "were not found."
    )


# ============================================================
# 11. LIMITACOES CIENTIFICAS
# ============================================================

st.header("6. Research Limitations")

st.write(
    """
    - The original research dataset contains 84 monthly observations
      from January 2019 to December 2025. Uploaded additional months
      are experimental and do not change historical validation results.

    - The outcome represents reported positive
      malaria test results, not population-based
      incidence rates.

    - Predictions depend on the availability
      and timeliness of surveillance data.

    - The model has not been prospectively
      validated in routine surveillance.

    - The model's historical performance was
      evaluated using rolling-origin validation
      over 24 months.

    - The contribution of climate variables
      was assessed through an exploratory
      Random Forest ablation analysis.

    - The prototype currently generates only
      one-month-ahead predictions.

    - Uploaded datasets must be verified against official source records;
      passing internal consistency checks does not establish accuracy.

    - Predictions should not be used as the
      sole basis for public-health decisions.
    """
)

# ============================================================
# 12. RELATORIO CIENTIFICO — PDF E WORD (PONTO 4)
# ============================================================


def construir_relatorio_cientifico(dados, historico, avaliacoes, caminho_metricas,
                                   origem, gerado_em):
    """Gera PDF e DOCX em memoria, sem alterar bases nem historicos."""
    from io import BytesIO
    from xml.sax.saxutils import escape
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from docx import Document
    from docx.shared import Inches, Pt
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        KeepTogether,
    )

    # Os documentos reproduzem apenas informacao efectivamente disponivel.
    serie = dados["Confirmed (HTZ + TDR)"].astype(float)
    periodo = f"{dados.index.min():%Y-%m} to {dados.index.max():%Y-%m}"
    origem_descricao = (
        "PUBLIC DEMONSTRATION - SYNTHETIC DATA ONLY. "
        "All values are artificially generated and do not "
        "represent observed malaria data from Quelimane."
    )
    metricas = None
    if caminho_metricas.exists():
        metricas = pd.read_csv(caminho_metricas)

    def figura(serie_x, titulo, eixo_y):
        fig, ax = plt.subplots(figsize=(9.1, 3.1))
        ax.plot(serie_x.index, serie_x.values, linewidth=1.6)
        ax.set_title(titulo)
        ax.set_ylabel(eixo_y)
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    figura_malaria = figura(serie, "Reported malaria-positive test results", "Positive results")
    figuras_clima = [
        ("Precipitation (mm)", "Monthly precipitation (mm)"),
        ("Mean temperature (°C)", "Mean temperature (degrees C)"),
        ("Relative humidity (%)", "Relative humidity (%)"),
    ]
    figura_clima = [
        (titulo, figura(dados[coluna].astype(float), titulo, titulo))
        for coluna, titulo in figuras_clima
    ]

    secoes = [
        ("1. Study scope and provenance", [
            "Location: Quelimane City, Mozambique.",
            f"Reporting period: {periodo} ({len(dados)} monthly observations).",
            f"Active dataset: {origem_descricao}.",
            "Outcome: reported positive malaria test results (HTZ + TDR); "
            "not unique patients or population-based incidence.",
        ]),
        ("2. Epidemiological surveillance", [
            f"Sum of reported positive test results: {int(serie.sum()):,}.",
            f"Latest month: {dados.index.max():%Y-%m}; "
            f"reported positive results: {int(serie.iloc[-1]):,}.",
            "The total is the sum of monthly reported positive test results, "
            "not an estimate of incident malaria cases.",
        ]),
        ("3. Climate monitoring", [
            "Climate variables shown: precipitation, mean temperature, and relative humidity.",
            "Descriptive climate patterns alone do not establish causal effects on malaria.",
        ]),
        ("4. Forecasts and evaluation", [
            "Model: RandomForestRegressor, one-month-ahead experimental forecast.",
            f"Archived forecasts: {len(historico)}; "
            f"registered evaluations: {len(avaliacoes)}.",
            "An archived forecast is not evidence of prospective validation. "
            "The generation timestamp, target month, observed-data provenance, "
            "and evaluation type must be checked independently.",
        ]),
        ("5. Model performance", [
            "Existing model comparison files, when present, describe historical "
            "rolling-origin evaluation (January 2024 to December 2025), "
            "not prospective operational performance.",
        ]),
        ("6. Scientific limitations and interpretation", [
            "This is an experimental research prototype and is not prospectively validated.",
            "Uploaded observations may be simulated or unverified. Internal "
            "consistency checks do not establish source authenticity.",
            "Historical model comparisons are retrospective. "
            "Do not use this report as a stand-alone public-health alert.",
            "No causal relationship between climate and malaria is claimed.",
        ]),
    ]

    def limpar_linhas(tabela, max_linhas=12):
        if tabela is None or tabela.empty:
            return []
        # Evitar relatorios excessivos e identificadores internos desnecessarios.
        return [list(map(str, linha)) for linha in
                tabela.head(max_linhas).fillna("").itertuples(index=False, name=None)]

    # Apresentacao comum aos dois formatos; valores originais permanecem intactos.
    def tabela_formatada(tabela, campos, nomes):
        saida = [nomes]
        for _, reg in tabela[campos].tail(10).iloc[::-1].iterrows():
            linha = []
            for campo in campos:
                valor = str(reg[campo])
                if campo == "generated_at_utc":
                    data = pd.to_datetime(valor, utc=True, errors="coerce")
                    valor = data.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(data) else valor
                elif campo in ("predicted_positive_results", "absolute_error",
                               "observed_positive_results", "percentage_error",
                               "MAE", "RMSE"):
                    numero = pd.to_numeric(valor, errors="coerce")
                    if pd.notna(numero):
                        valor = f"{float(numero):,.2f}"
                elif campo == "forecast_classification":
                    valor = valor.replace("experimental_unverified", "Experimental / unverified")
                elif campo == "verification_status":
                    valor = valor.replace("_", " ")
                linha.append(valor)
            saida.append(linha)
        return saida

    campos_prev = ["forecast_month", "predicted_positive_results",
                   "generated_at_utc", "forecast_classification"]
    nomes_prev = ["Month", "Forecast", "Generated (UTC)", "Classification"]
    campos_eval = ["forecast_month", "absolute_error", "evaluation_type",
                   "verification_status"]
    nomes_eval = ["Month", "Absolute error", "Evaluation type", "Verification"]

    # WORD — impedir que uma linha da tabela seja dividida entre paginas.
    # Nao altera os valores nem o conteudo cientifico.
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    def proteger_paginacao_tabela(tabela):
        for indice, linha in enumerate(tabela.rows):
            propriedades = linha._tr.get_or_add_trPr()
            if propriedades.find(qn("w:cantSplit")) is None:
                propriedades.append(OxmlElement("w:cantSplit"))
            # Manter o cabecalho junto da primeira linha de dados.
            if indice == 0 and len(tabela.rows) > 1:
                for celula in linha.cells:
                    for paragrafo in celula.paragraphs:
                        paragrafo.paragraph_format.keep_with_next = True

    # WORD
    doc = Document()
    sec = doc.sections[0]
    sec.header.paragraphs[0].text = "Quelimane | Malaria and Climate Research Prototype"
    doc.add_heading("Scientific Monitoring and Forecast Report", 0)
    doc.add_paragraph(f"Generated (UTC): {gerado_em}")
    doc.add_paragraph("RESEARCH USE ONLY — DATA PROVENANCE NOT INDEPENDENTLY VERIFIED")
    for i, (titulo, paragrafos) in enumerate(secoes):
        doc.add_heading(titulo, level=1)
        for paragrafo in paragrafos:
            doc.add_paragraph(paragrafo)
        if i == 1:
            doc.add_picture(BytesIO(figura_malaria), width=Inches(6.1))
        if i == 2:
            for titulo_clima, img in figura_clima:
                doc.add_picture(BytesIO(img), width=Inches(6.1))
        if i == 3 and not historico.empty:
            doc.add_heading("Archived forecasts (latest 10)", level=2)
            linhas = tabela_formatada(historico, campos_prev, nomes_prev)
            t = doc.add_table(rows=1, cols=len(nomes_prev))
            t.style = "Light Shading Accent 1"
            for cel, nome in zip(t.rows[0].cells, linhas[0]):
                cel.text = nome
            for linha in linhas[1:]:
                for cel, valor in zip(t.add_row().cells, linha):
                    cel.text = valor
            proteger_paginacao_tabela(t)
        if i == 3 and not avaliacoes.empty:
            doc.add_heading("Registered evaluations (latest 10)", level=2)
            linhas = tabela_formatada(avaliacoes, campos_eval, nomes_eval)
            t = doc.add_table(rows=1, cols=len(nomes_eval))
            t.style = "Light Shading Accent 1"
            for cel, nome in zip(t.rows[0].cells, linhas[0]):
                cel.text = nome
            for linha in linhas[1:]:
                for cel, valor in zip(t.add_row().cells, linha):
                    cel.text = valor
            proteger_paginacao_tabela(t)
        if i == 4 and metricas is not None:
            doc.add_heading("Archived model-comparison metrics", level=2)
            doc.add_paragraph("Values from the existing model comparison CSV; rounded for display.")
            cols = list(metricas.columns)[:6]
            linhas = tabela_formatada(metricas.iloc[::-1], cols, cols)
            t = doc.add_table(rows=1, cols=len(cols))
            t.style = "Light Shading Accent 1"
            for cel, nome in zip(t.rows[0].cells, linhas[0]):
                cel.text = str(nome)
            for linha in linhas[1:]:
                for cel, valor in zip(t.add_row().cells, linha):
                    cel.text = valor
            proteger_paginacao_tabela(t)
    docx_buffer = BytesIO()
    doc.save(docx_buffer)

    # PDF: platypus, com tabelas de dimensoes limitadas e texto adaptavel.
    pdf_buffer = BytesIO()
    pdf_doc = SimpleDocTemplate(pdf_buffer, pagesize=(21*cm, 29.7*cm),
                                rightMargin=1.8*cm, leftMargin=1.8*cm,
                                topMargin=1.8*cm, bottomMargin=1.8*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportSmall", parent=styles["Normal"],
                              fontSize=7, leading=9))
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                              alignment=TA_CENTER, spaceAfter=14))
    story = [Paragraph("Scientific Monitoring and Forecast Report", styles["ReportTitle"]),
             Paragraph(f"Quelimane City, Mozambique | Generated UTC: {escape(gerado_em)}",
                       styles["Normal"]),
             Spacer(1, 10),
             Paragraph("RESEARCH USE ONLY - DATA PROVENANCE NOT INDEPENDENTLY VERIFIED",
                       styles["Heading3"])]

    def pdf_tabela(dados_tabela):
        if not dados_tabela:
            return
        largura = (21*cm - 3.6*cm) / len(dados_tabela[0])
        celulas = [
            [Paragraph(escape(str(c)), styles["ReportSmall"]) for c in linha]
            for linha in dados_tabela
        ]
        tab = Table(celulas, colWidths=[largura]*len(dados_tabela[0]), repeatRows=1,
                    hAlign="LEFT")
        tab.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E7EDF4")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("GRID", (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(tab)
        story.append(Spacer(1, 12))

    def pdf_figura(img):
        imagem = Image(BytesIO(img), width=16.7*cm, height=5.7*cm)
        story.append(imagem)
        story.append(Spacer(1, 8))

    for i, (titulo, paragrafos) in enumerate(secoes):
        story.append(Spacer(1, 12))
        story.append(Paragraph(escape(titulo), styles["Heading2"]))
        for paragrafo in paragrafos:
            story.append(Paragraph(escape(paragrafo), styles["Normal"]))
            story.append(Spacer(1, 5))
        if i == 1:
            pdf_figura(figura_malaria)
        if i == 2:
            for _, img in figura_clima:
                pdf_figura(img)
        if i == 3 and not historico.empty:
            story.append(Paragraph("Archived forecasts (latest 10)", styles["Heading3"]))
            pdf_tabela(tabela_formatada(historico, campos_prev, nomes_prev))
        if i == 3 and not avaliacoes.empty:
            story.append(Paragraph("Registered evaluations (latest 10)", styles["Heading3"]))
            pdf_tabela(tabela_formatada(avaliacoes, campos_eval, nomes_eval))
        if i == 4 and metricas is not None:
            story.append(Paragraph("Archived model-comparison metrics", styles["Heading3"]))
            cols = list(metricas.columns)[:6]
            pdf_tabela(tabela_formatada(metricas.iloc[::-1], cols, cols))
    pdf_doc.build(story)
    return pdf_buffer.getvalue(), docx_buffer.getvalue()


st.header("7. Scientific Reports")
st.caption(
    "Generate a research report in PDF and editable Word format. "
    "Reports are prepared from the active dataset and archived records; "
    "generating or downloading a report does not modify source files."
)
if ficheiro_actualizado is not None:
    st.warning(
        "An uploaded dataset is active. Its provenance is unverified and it "
        "may contain simulated observations. The report will be marked accordingly."
    )
else:
    st.warning("PUBLIC DEMONSTRATION - SYNTHETIC DATA ONLY. No original epidemiological observations are included.")

assinatura_relatorio = calcular_fingerprint_dataset(df)
if st.button("Prepare PDF and Word Reports", key="prepare_scientific_reports"):
    try:
        with st.spinner("Preparing scientific reports..."):
            momento = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            pdf_gerado, docx_gerado = construir_relatorio_cientifico(
                df, carregar_historico(), carregar_avaliacoes(), ficheiro_metricas,
                ficheiro_actualizado is not None, momento,
            )
            st.session_state["scientific_report_files"] = {
                "fingerprint": assinatura_relatorio,
                "pdf": pdf_gerado,
                "docx": docx_gerado,
            }
        st.success("Reports prepared. Download using the buttons below.")
    except Exception as erro_relatorio:
        st.error(f"Could not generate scientific reports: {erro_relatorio}")

ficheiros_relatorio = st.session_state.get("scientific_report_files")
if ficheiros_relatorio and ficheiros_relatorio["fingerprint"] == assinatura_relatorio:
    c_pdf, c_docx = st.columns(2)
    with c_pdf:
        st.download_button(
            "Download Scientific Report (PDF)",
            data=ficheiros_relatorio["pdf"],
            file_name="Quelimane_scientific_report.pdf",
            mime="application/pdf", key="download_scientific_pdf",
        )
    with c_docx:
        st.download_button(
            "Download Scientific Report (Word)",
            data=ficheiros_relatorio["docx"],
            file_name="Quelimane_scientific_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_scientific_docx",
        )
    st.caption(
        "Files are downloaded to your browser's download location. "
        "The scientific_reports directory is reserved for manually archived reports."
    )


st.success(
    "Experimental forecasting prototype ready."
)
