# Malaria Forecasting and Climate Monitoring — Quelimane, Mozambique

## Overview

This repository contains an experimental Python/Streamlit research prototype for monitoring climate variables and reported malaria-positive test results and exploring monthly forecasting in Quelimane City, Mozambique.

**Research-use warning:** The forecasts concern reported malaria-positive test results, **not** population-based malaria incidence rates or necessarily unique malaria cases. The system has **not** been prospectively validated and is not intended for clinical or operational public-health decision-making.

## Study area and data

- **Study area:** Quelimane City, Zambézia Province, Mozambique.
- **Epidemiological series:** January 2019–December 2025 (monthly).
- **Climate reference series:** 1991–2025, obtained from NASA POWER.
- **Climate variables:** precipitation, mean/minimum/maximum air temperature, and relative humidity.

The provenance, completeness, accuracy, and permission for public redistribution of the epidemiological dataset require independent confirmation. The `data/` directory is deliberately excluded from the source-code distribution.

## Methods

The interactive application generates experimental monthly forecasts using a `RandomForestRegressor` and presents historical model comparisons. The research workflow includes rolling-origin evaluation over January 2024–December 2025 and analyses of climate–malaria relationships. Historical error estimates do not demonstrate prospective predictive performance; climate associations must not be interpreted as causal effects.

## Features

- Epidemiological and climate monitoring and visualizations.
- Experimental monthly forecasts and forecast-history registry.
- Historical forecasting-model comparisons.
- Controlled monthly dataset-update registration.
- Comparison of archived forecasts with subsequently supplied observations, with verification-status safeguards.
- Downloadable scientific reports in PDF and editable Word formats.

The application keeps local CSV registries for dataset updates, forecasts, and evaluations. These records are not distributed publicly.

## Requirements

Developed with **Python 3.13.17**. Direct application dependencies are listed in `requirements.txt`.

## Local setup

From the project directory, create and activate a Python virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Required local data

The current version expects these files in `data/malaria/`:

- `Quelimane_clima_malaria_integrado_2019_2025(1).xlsx`
- `comparacao_todos_modelos_2024_2025.csv`
- `previsoes_random_forest_2024_2025.csv`

These files **are not included** in a public source-code release. Authorized users must supply compatible local files in the expected locations. Without them, parts of the application may fail to load or be unavailable. The public source code is therefore **not yet a standalone runnable demonstration**; a separately approved demonstration dataset and corresponding code adjustments would be needed for that purpose.

### Start the application

```powershell
python -m streamlit run .\prototipo_malaria_quelimane.py
```

Open the local URL displayed by Streamlit.

## Scientific and data-governance limitations

1. No prospective validation has been established.
2. Epidemiological data have not been independently verified for public release.
3. Forecast targets are reported positive test results, not population incidence.
4. Historical validation does not guarantee future forecast accuracy.
5. Climate associations do not establish causation.
6. Uploaded or simulated test datasets must not be presented as official observations.
7. Public deployment requires a separate privacy, security, data-permission, and operational review.

## Publication status

**Pre-release experimental research software.** No public release, data-sharing authorization, or production deployment is implied by this README. Licensing, authorship, institutional acknowledgments, citation instructions, and repository publication settings should be confirmed before dissemination.
