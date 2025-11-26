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
from scipy.stats import pearsonr, spearmanr, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")


def format_df(
    df, 
    exog=False, 
    add_outlier=True,
    use_log=False,
    use_boxcox=False,
    target_mode="absolute",   # "absolute", "diff1", "diff2", "ratio", "relative"
):
    df = df.copy()

    # Selección de columnas si no hay exógenas
    if not exog:
        df = df[["CASES"]]

    original_cases = df["CASES"].copy()

    # 1. Transformación de la serie
    if use_log:
        df["CASES"] = np.log1p(df["CASES"])

    if use_boxcox:
        df["CASES"], _ = boxcox(df["CASES"] + 1)

    # 2. Construcción del objetivo
    if target_mode == "absolute":
        df["CASES"] = df["CASES"]

    elif target_mode == "diff1":
        df["CASES"] = df["CASES"].diff()
    
    elif target_mode == "diff2":
        df["CASES"] = df["CASES"].diff().diff()

    elif target_mode == "ratio":
        df["CASES"] = df["CASES"] / (df["CASES"].shift(1) + 1e-6)

    elif target_mode == "relative":
        df["CASES"] = (df["CASES"] - df["CASES"].shift(1)) / (df["CASES"].shift(1) + 1e-6)

    # 3. Cálculo de OUTLIERS (en el espacio original)
    if add_outlier:
        Q1 = original_cases.quantile(0.25)
        Q3 = original_cases.quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        df["OUTLIER"] = (original_cases.loc[df.index] > threshold).astype(int)

    # Eliminar NaNs generados por shifts
    df = df.dropna().copy()

    return df


def inverse_transform_target(
    y_pred,
    original_series,
    use_log=False,
    use_boxcox=False,
    target_mode="absolute",
    boxcox_lambda=None
):
    y_pred = np.array(y_pred).flatten()

    # 1. Revert target transform

    # Absolute (no transformación)
    if target_mode == "absolute":
        restored = y_pred.copy()

    # Primera diferencia
    elif target_mode == "diff1":
        restored = original_series[-1] + np.cumsum(y_pred)

    # Segunda diferencia
    elif target_mode == "diff2":
        diff1 = np.cumsum(y_pred)
        restored = original_series[-1] + np.cumsum(diff1)

    # Ratio: y_t = y_(t-1) * ratio
    elif target_mode == "ratio":
        restored = []
        last = original_series[-1]
        for r in y_pred:
            nxt = last * r
            restored.append(nxt)
            last = nxt
        restored = np.array(restored)

    # Relative: (y_t - y_(t-1))/y_(t-1)
    elif target_mode == "relative":
        restored = []
        last = original_series[-1]
        for rel in y_pred:
            nxt = last * (1 + rel)
            restored.append(nxt)
            last = nxt
        restored = np.array(restored)

    else:
        raise ValueError("target_mode inválido.")

    # 2. Revert log or boxcox
    if use_boxcox:
        restored = inv_boxcox(restored, boxcox_lambda)

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
    plt.title("Actual vs Predicted Over Time (LSTM)")
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
    plt.title("Error Distribution (LSTM)")
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
    plt.title("Actual vs Predicted (LSTM)")
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
    plt.title("Residuals Over Time (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_history(history, outdir: Path):
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(outdir, index=False)


def plot_training_history(history, outdir: Path):
    hist_df = pd.DataFrame(history.history)
    
    # Detectar automáticamente la métrica de loss
    loss_name = history.model.loss if hasattr(history.model, 'loss') else 'loss'
    loss_name_str = loss_name.upper() if isinstance(loss_name, str) else str(loss_name)
    
    plt.figure(figsize=(10,6))
    plt.plot(hist_df["loss"], label="Train Loss", linewidth=2)
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="Validation Loss")
    
    plt.title(f"LSTM Training History ({loss_name_str} loss)")
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss ({loss_name_str})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(outdir, dpi=150, bbox_inches="tight")
    plt.close()


def save_model(model_fit, outpath: Path):
    with open(outpath, "wb") as f:
        pickle.dump(model_fit, f)

def save_metrics_csv(metrics_dict: dict, outpath: Path):
    pd.DataFrame([metrics_dict]).to_csv(outpath, index=False)

def save_true_pred(y_true, y_pred, outpath: Path):
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
        
    df_tp = pd.DataFrame({"Actual": y_true.values, "Predicted": y_pred.values}, index=y_true.index)
    df_tp.to_pickle(outpath)

def save_pickle(obj, outpath: Path):
    with open(outpath, "wb") as f:
        pickle.dump(obj, f)


def reshape_data(X, y, timesteps):
    # Convertimos y en columna y la unimos a X
    y_col = np.array(y).reshape(-1, 1)
    X_full = np.hstack((X, y_col))

    Xs, ys = [], []

    for i in range(len(X_full) - timesteps):
        Xs.append(X_full[i:i+timesteps])
        ys.append(y[i+timesteps])

    return np.array(Xs), np.array(ys)


def grid_search(params, build_model, X_train, y_train, X_test, y_test, train_df, scaler_y, lstm_units_tuples, dropout_rates_tuples, learning_rates, timesteps_values, epochs, batch_size):
    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()
    X_test_orig = X_test.copy()
    y_test_orig = y_test.copy()

    best_rmse = np.inf
    best_config = None
    best_model = None
    best_history = None

    results = []

    print("Buscando mejores parámetros LSTM (lstm_units, dropout_rates, learning_rate, timesteps)...\n")

    total = len(lstm_units_tuples) * len(dropout_rates_tuples) * len(learning_rates) * len(timesteps_values)
    count = 1

    for lstm_units, dropout_rates, learning_rate, timesteps in itertools.product(lstm_units_tuples, dropout_rates_tuples, learning_rates, timesteps_values):
        try:
            print(f"[{count}/{total}] Starting LSTM (lstm_units = {lstm_units}, dropout_rates = {dropout_rates}, learning_rate = {learning_rate}, timesteps = {timesteps})...")

            X_train, y_train, X_test, y_test = X_train_orig.copy(), y_train_orig.copy(), X_test_orig.copy(), y_test_orig.copy()
            X_train, y_train, X_test, y_test = X_train[max(timesteps_values)-timesteps:], y_train[max(timesteps_values)-timesteps:], X_test[max(timesteps_values)-timesteps:], y_test[max(timesteps_values)-timesteps:]
            X_train, y_train = reshape_data(X_train, y_train, timesteps=timesteps)
            X_test, y_test = reshape_data(X_test, y_test, timesteps=timesteps)

            model = build_model((X_train.shape[1], X_train.shape[2]), lstm_units, dropout_rates, learning_rate)
            es = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es], shuffle=False)

            # Predicciones
            y_pred_scaled = model.predict(X_test)
            y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled).flatten()
            y_true_rescaled = scaler_y.inverse_transform(y_test).flatten()

            y_pred = inverse_transform_target(
                y_pred_rescaled,
                train_df["CASES"].values,  
                use_log=params["use_log"],
                use_boxcox=params["use_boxcox"],
                target_mode=params["target_mode"],
            )
            y_true = inverse_transform_target(
                y_true_rescaled,
                train_df["CASES"].values,
                use_log=params["use_log"],
                use_boxcox=params["use_boxcox"],
                target_mode=params["target_mode"],
            )

            # Evaluar con métricas extendidas
            metrics = evaluate_model(y_true, y_pred)

            # Añadir info del modelo + métricas
            results.append({
                "lstm_units": lstm_units,
                "dropout_rates": dropout_rates,
                "learning_rate": learning_rate,
                "timesteps": timesteps,
                **metrics
            })

            print(f"LSTM (lstm_units = {lstm_units}, dropout_rates = {dropout_rates}, learning_rate = {learning_rate}, timesteps = {timesteps}) → RMSE = {metrics['RMSE']:.4f}")

            # Guardar el mejor modelo
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_config = {
                    "lstm_units": lstm_units,
                    "dropout_rates": dropout_rates,
                    "learning_rate": learning_rate,
                    "timesteps": timesteps
                }
                best_model = model
                best_history = history
                best_y_true = y_true.copy()
                best_y_pred = y_pred.copy()
                print("✅ Nuevo mejor modelo encontrado")

        except Exception as e:
            print(f"❌ Error con (lstm_units = {lstm_units}, dropout_rates = {dropout_rates}, learning_rate = {learning_rate}, timesteps = {timesteps}): {e}")
        
        count += 1
        print("\n")

    return best_rmse, best_model, best_config, best_history, best_y_true, best_y_pred, results


def train_final_model(params, build_model, X_train, y_train, X_test, y_test, train_df, test_df, scaler_y, best_rmse, best_config, epochs, batch_size):
    print("Mejor modelo encontrado:")
    print(f"     LSTM Units: {best_config['lstm_units']} | Dropout: {best_config['dropout_rates']} | "
          f"     Learning Rate: {best_config['learning_rate']} | Timesteps: {best_config['timesteps']} | "
          f"     RMSE (val): {best_rmse:.4f}\n\n")
    print("Reentrenar modelo LSTM con la mejor configuración y probando en datos de test...")
    print("Ejecutando Backtesting con 3 folds (61 registros cada uno)...\n")

    T = best_config["timesteps"]
    F = 61  # tamaño del fold
    total_test = len(X_test[best_config["timesteps"]:])
    assert total_test == 183, f"ERROR: El test no tiene 183 registros, tiene {total_test}"

    folds_X = []
    folds_y = []

    for i in range(3):
        e_start = i * F
        e_end = (i + 1) * F
        f_start = e_start
        f_end = e_end + T
        folds_X.append(X_test[f_start:f_end])
        folds_y.append(y_test[f_start:f_end])
    models = []
    histories = []
    all_y_true = []
    all_y_pred = []

    for i in range(3):
        print(f"\n============== 📘 Fold {i+1}/3 ==============\n")

        # Test actual
        X_test_fold = folds_X[i]
        y_test_fold = folds_y[i]

        X_train_bt, y_train_bt = reshape_data(X_train, y_train, timesteps=best_config['timesteps'])
        X_test_bt, y_test_bt = reshape_data(X_test_fold, y_test_fold, timesteps=best_config['timesteps'])

        model = build_model((X_train_bt.shape[1], X_train_bt.shape[2]), **{k: best_config[k] for k in ["lstm_units","dropout_rates","learning_rate"]})
        print(model.summary())

        es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        history = model.fit(X_train_bt, y_train_bt, validation_data=(X_test_bt, y_test_bt), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es], shuffle=False)

        y_pred_scaled = model.predict(X_test_bt)
        y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true_rescaled = scaler_y.inverse_transform(y_test_bt).flatten()

        y_pred = inverse_transform_target(
            y_pred_rescaled,
            train_df["CASES"].values,
            use_log=params["use_log"],
            use_boxcox=params["use_boxcox"],
            target_mode=params["target_mode"],
        )
        y_true = inverse_transform_target(
            y_true_rescaled,
            train_df["CASES"].values,
            use_log=params["use_log"],
            use_boxcox=params["use_boxcox"],
            target_mode=params["target_mode"],
        )

        # Guardar resultados
        models.append(model)
        histories.append(history)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # Expandir train para el siguiente fold
        X_train = np.vstack([X_train, X_test_fold[T:]])
        y_train = np.vstack([y_train, y_test_fold[T:]])
        train_df = pd.concat([train_df, test_df.iloc[T + i*F:T + (i+1)*F]]) 

    return models[-1], histories[-1], np.array(all_y_true), np.array(all_y_pred)


def save_val_data(directory, X_train, X_test, val_df, best_config, best_history, best_y_true, best_y_pred, results):
    # Guardar validación
    ensure_dir(directory)

    # Guardar grid search
    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    results_df.to_csv(f"{directory}/results_table.csv", index=False)

    # Guardar resumen
    save_training_history(best_history, f"{directory}/best_history.csv")
    plot_training_history(best_history, f"{directory}/best_history.png")

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


def save_test_data(directory, X_train, X_test, test_df, history, y_true, y_pred):
    # Guardar test
    ensure_dir(directory)

    # Guardar resumen
    save_training_history(history, f"{directory}/history.csv")
    plot_training_history(history, f"{directory}/history.png")

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