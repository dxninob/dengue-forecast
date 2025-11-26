import os
from pathlib import Path
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import pearsonr, spearmanr, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox

import warnings
warnings.filterwarnings("ignore")


def format_df(
    df, 
    exog=False, 
    add_outlier=True,
    use_log=False,
    target_mode="absolute",   # "absolute", "ratio", "relative"
):
    df = df.copy()

    if not exog:
        df = df[["CASES"]]

    original_cases = df["CASES"].copy()

    if use_log:
        df["CASES"] = np.log1p(df["CASES"])

    if target_mode == "absolute":
        df["CASES"] = df["CASES"]
    elif target_mode == "ratio":
        df["CASES"] = df["CASES"] / (df["CASES"].shift(1) + 1e-6)
    elif target_mode == "relative":
        df["CASES"] = (df["CASES"] - df["CASES"].shift(1)) / (df["CASES"].shift(1) + 1e-6)

    if add_outlier:
        Q1 = original_cases.quantile(0.25)
        Q3 = original_cases.quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        df["OUTLIER"] = (original_cases.loc[df.index] > threshold).astype(int)

    df = df.dropna().copy()

    return df


def inverse_transform_target(
    y_pred,
    original_series,
    use_log=False,
    target_mode="absolute",
):
    y_pred = np.array(y_pred).flatten()

    if target_mode == "absolute":
        restored = y_pred.copy()
    elif target_mode == "ratio":
        restored = []
        last = original_series[-1]
        for r in y_pred:
            nxt = last * r
            restored.append(nxt)
            last = nxt
        restored = np.array(restored)

    elif target_mode == "relative":
        restored = []
        last = original_series[-1]
        for rel in y_pred:
            nxt = last * (1 + rel)
            restored.append(nxt)
            last = nxt
        restored = np.array(restored)

    if use_log:
        restored = np.expm1(restored)

    return restored


def split_by_date(data, validation_date, test_date):
    validation_date = pd.to_datetime(validation_date)
    test_date = pd.to_datetime(test_date)

    train_val_df = data[data.index < test_date]
    train_df = data[data.index < validation_date]
    validation_df = data[(data.index >= validation_date) & (data.index < test_date)]
    test_df = data[data.index >= test_date]

    return train_val_df, train_df, validation_df, test_df


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom[denom == 0] = 1e-8
    return np.mean(np.abs(y_true - y_pred) / denom)


def evaluate_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calcular errores básicos
    residuals = y_true - y_pred
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # SMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom[denom == 0] = 1e-8
    smape_val = np.mean(np.abs(y_true - y_pred) / denom) * 100

    # Correlaciones
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    # Errores adicionales
    mean_error = np.mean(y_pred - y_true)
    safe_y_true = np.where(y_true == 0, 1e-8, y_true)
    mean_percent_error = np.nanmean((y_pred - y_true) / safe_y_true) * 100
    error_std = np.std(residuals)

    # Test Ljung-Box
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_stat = lb_test["lb_stat"].values[0]
    lb_pvalue = lb_test["lb_pvalue"].values[0]

    # Test Shapiro-Wilk
    stat, p_value = shapiro(residuals)

    # Consolidar métricas
    metrics = {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "SMAPE (%)": smape_val,
        "Pearson Corr": pearson_corr,
        "Spearman Corr": spearman_corr,
        "Mean Error (Bias)": mean_error,
        "Mean % Error (MPE)": mean_percent_error,
        "Error STD": error_std,
        "Ljung-Box Stat (lag10)": lb_stat,
        "Ljung-Box p-value (lag10)": lb_pvalue,
        "Shapiro-Wilk Stat": stat,
        "Shapiro-Wilk p-value": p_value,
    }

    # Mostrar resultados
    print("   📊 --- Model Evaluation ---")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R2: {r2:.4f}")
    print(f"   SMAPE: {smape_val:.2f}%")
    print(f"   Pearson Corr: {pearson_corr:.4f}")
    print(f"   Spearman Corr: {spearman_corr:.4f}")
    print(f"   Mean Error (Bias): {mean_error:.4f}")
    print(f"   Mean % Error (MPE): {mean_percent_error:.2f}%")
    print(f"   Error Standard Deviation: {error_std:.4f}")
    print(f"   Ljung-Box Stat (lag 10): {lb_stat:.4f}")
    print(f"   Ljung-Box p-value (lag 10): {lb_pvalue:.4f}")
    print(f"   Shapiro-Wilk Stat (normality): {stat:.4f}")
    print(f"   Shapiro-Wilk p-value (normality): {p_value:.4f}")

    return metrics


def ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)


def format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.xticks(rotation=90)


def plot_predictions(dates, y_true, y_pred, outpath: Path):
    plt.figure(figsize=(12,6))
    plt.plot(dates, y_true, label="Actual", alpha=0.7)
    plt.plot(dates, y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted Over Time (SARIMAX)")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_distribution(y_true, y_pred, outpath: Path):
    errors = y_true - y_pred
    plt.figure(figsize=(10,6))
    plt.hist(errors, bins=50, color="steelblue", edgecolor="black")
    plt.title("Error Distribution (SARIMAX)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter(y_true, y_pred, outpath: Path):
    plt.figure(figsize=(8,8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    minv = min(min(y_true), min(y_pred))
    maxv = max(max(y_true), max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (SARIMAX)")
    plt.legend(["y = x"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals_over_time(dates, y_true, y_pred, outpath: Path):
    errors = y_true - y_pred
    plt.figure(figsize=(12,6))
    plt.plot(dates, errors, label="Residual (y_true - y_pred)", color="orange")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals Over Time (SARIMAX)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def save_model(model_fit, outpath: Path):
    with open(outpath, "wb") as f:
        pickle.dump(model_fit, f)

def save_text(text: str, outpath: Path):
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)

def save_metrics_csv(metrics_dict: dict, outpath: Path):
    pd.DataFrame([metrics_dict]).to_csv(outpath, index=False)

def save_true_pred(y_true, y_pred, outpath: Path):
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
        
    df_tp = pd.DataFrame({"Actual": y_true.values, "Predicted": y_pred.values}, index=y_true.index)
    df_tp.to_pickle(outpath)


def save_val_data(directory, X_train, X_test, val_df, best_y_true, best_y_pred, results):
    # Guardar validación
    ensure_dir(directory)

    # Guardar grid search
    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    results_df.to_csv(f"{directory}/results_table.csv", index=False)

    # Guardar métricas
    best_metrics = evaluate_model(best_y_true, best_y_pred)
    save_metrics_csv(best_metrics, f"{directory}/best_metrics.csv")

    # Guardar gráficos
    plot_predictions(val_df.index[:len(best_y_true)], best_y_true, best_y_pred, f"{directory}/predictions.png")
    plot_error_distribution(best_y_true, best_y_pred, f"{directory}/error_distribution.png")
    plot_scatter(best_y_true, best_y_pred, f"{directory}/scatter.png")
    plot_residuals_over_time(val_df.index[:len(best_y_true)], best_y_true, best_y_pred, f"{directory}/residuals.png")

    # Guardar true/pred
    save_true_pred(best_y_true, best_y_pred, f"{directory}/best_true_pred.pkl")


def save_test_data(directory, X_train, X_test, test_df, y_true, y_pred):
    # Guardar test
    ensure_dir(directory)

    # Guardar métricas
    test_metrics = evaluate_model(y_true, y_pred)
    save_metrics_csv(test_metrics, f"{directory}/metrics.csv")

    # Guardar gráficos
    plot_predictions(test_df.index[:len(y_true)], y_true, y_pred, f"{directory}/predictions.png")
    plot_error_distribution(y_true, y_pred, f"{directory}/error_distribution.png")
    plot_scatter(y_true, y_pred, f"{directory}/scatter.png")
    plot_residuals_over_time(test_df.index[:len(y_true)], y_true, y_pred, f"{directory}/residuals.png")

    # Guardar true/pred
    save_true_pred(y_true, y_pred, f"{directory}/true_pred.pkl")