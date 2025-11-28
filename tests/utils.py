import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")


def format_df(
    df, 
    exog=False, 
    add_outlier=True,
    use_log=False,
    target_mode="absolute",
):
    df = df.copy()

    if not exog:
        df = df[["CASES"]]

    original_cases = df["CASES"].copy()

    if use_log:
        df["CASES"] = np.log1p(df["CASES"])

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
    elif target_mode == "diff1":
        restored = original_series[-1] + np.cumsum(y_pred)
    elif target_mode == "diff2":
        diff1 = np.cumsum(y_pred)
        restored = original_series[-1] + np.cumsum(diff1)
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


def split_by_date(data, start_date, test_date, end_date):
    start_date = pd.to_datetime(start_date)
    test_date = pd.to_datetime(test_date)

    train_df = data[(data.index >= start_date) & (data.index < test_date)]
    test_df = data[(data.index >= test_date) & (data.index < end_date)]

    return train_df, test_df


def evaluate_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calcular errores básicos
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    # Consolidar métricas
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
    }

    # Mostrar resultados
    print("   📊 --- Model Evaluation ---")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")

    y_naive = y_true[:-1]        # sacamos el último valor
    y_true_comp = y_true[1:]     # alineamos ambas series
    rmse_naive = mean_squared_error(y_true_comp, y_naive) ** 0.5

    if rmse <= 0.5 * rmse_naive:
        print("   ✔️ Modelo candidato (RMSE ≤ 50% del naive)")
    else:
        print("   ❌ Modelo NO candidato (no mejora lo esperado sobre naive)")

    return metrics


def format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.xticks(rotation=90)


def plot_predictions(dates, y_true, y_pred):
    plt.figure(figsize=(9,4))
    plt.plot(dates, y_true, label="Actual", alpha=0.7)
    plt.plot(dates, y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    format_time_axis(plt.gca())
    plt.tight_layout()
    plt.show()


def reshape_data(X, y, timesteps):
    y_col = np.array(y).reshape(-1, 1)
    X_full = np.hstack((X, y_col))

    Xs, ys = [], []

    for i in range(len(X_full) - timesteps):
        Xs.append(X_full[i:i+timesteps])
        ys.append(y[i+timesteps])

    return np.array(Xs), np.array(ys)


def train_final_model(params, build_model, X_train, y_train, X_test, y_test, train_df, test_df, scaler_y, timesteps, epochs, batch_size, loss):

    X_train, y_train = reshape_data(X_train, y_train, timesteps=timesteps)
    X_test, y_test = reshape_data(X_test, y_test, timesteps=timesteps)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer=Adam(), loss=loss)
    # print(model.summary())

    es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es], shuffle=False)

    y_pred_scaled = model.predict(X_test)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true_rescaled = scaler_y.inverse_transform(y_test).flatten()

    y_pred = inverse_transform_target(
        y_pred_rescaled,
        train_df["CASES"].values,
        use_log=params["use_log"],
        target_mode=params["target_mode"],
    )
    y_true = inverse_transform_target(
        y_true_rescaled,
        train_df["CASES"].values,
        use_log=params["use_log"],
        target_mode=params["target_mode"],
    )

    y_true = pd.Series(y_true, index=test_df[timesteps:].index)
    y_pred = pd.Series(y_pred, index=test_df[timesteps:].index)

    evaluate_model(y_true, y_pred)
    plot_predictions(test_df.index[:len(y_true)], y_true, y_pred)
