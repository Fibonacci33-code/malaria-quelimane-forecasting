from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base = Path("public_demo/data/malaria")
origem = base / "Quelimane_clima_malaria_integrado_2019_2025(1).xlsx"

df = pd.read_excel(origem)
df["date"] = pd.to_datetime(
    dict(year=df["Year"], month=df["Month"], day=1)
)
df = df.sort_values("date").set_index("date")

clima = [
    "Precipitation (mm)",
    "Mean temperature (" + chr(176) + "C)",
    "Relative humidity (%)",
]

y = df["Confirmed (HTZ + TDR)"].astype(float)

features = pd.DataFrame(index=df.index)
features["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
features["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

for lag in [1, 2, 3, 12]:
    features[f"malaria_lag_{lag}"] = y.shift(lag)

for coluna in clima:
    for lag in [1, 2, 3]:
        features[f"{coluna}_lag_{lag}"] = df[coluna].shift(lag)

dataset = features.copy()
dataset["target"] = y
dataset = dataset.dropna()

resultados = []

for data_teste in dataset.loc["2024-01":"2025-12"].index:
    treino = dataset.loc[dataset.index < data_teste]
    teste = dataset.loc[[data_teste]]

    if len(treino) < 24:
        raise ValueError("Histórico insuficiente para avaliação temporal.")

    modelo = RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(
        treino.drop(columns="target"),
        treino["target"],
    )

    previsto = float(
        modelo.predict(teste.drop(columns="target"))[0]
    )

    observado = float(teste["target"].iloc[0])
    anterior = float(y.loc[data_teste - pd.DateOffset(months=12)])

    resultados.append({
        "Month": data_teste.strftime("%Y-%m"),
        "Observed": observado,
        "Random_Forest": previsto,
        "Seasonal_Naive": anterior,
    })

previsoes = pd.DataFrame(resultados)

metricas = []
for nome, coluna in [
    ("Random Forest", "Random_Forest"),
    ("Seasonal Naive", "Seasonal_Naive"),
]:
    metricas.append({
        "Model": nome,
        "MAE": mean_absolute_error(
            previsoes["Observed"], previsoes[coluna]
        ),
        "RMSE": np.sqrt(mean_squared_error(
            previsoes["Observed"], previsoes[coluna]
        )),
        "Evaluation_Months": len(previsoes),
    })

pd.DataFrame(metricas).to_csv(
    base / "comparacao_todos_modelos_2024_2025.csv",
    index=False,
)

previsoes.to_csv(
    base / "previsoes_random_forest_2024_2025.csv",
    index=False,
)

print("Avaliação demonstrativa concluída.")
print("Meses avaliados:", len(previsoes))
print(pd.DataFrame(metricas).to_string(index=False))
print("DADOS EXCLUSIVAMENTE SINTÉTICOS.")
