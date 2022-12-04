import datetime
import json
import pickle
import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.wandb import *
from fastmsc.utils import *
from more_itertools import windowed
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tsai.all import *
from tsai.data.tabular import EarlyStoppingCallback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPORTS_DIR = Path("./exports")
FIGURES_DIR = Path("./figures")

# Data


def load_stock_price_dataset(path):
    return pd.read_csv(
        str(path),
        index_col="datetime",
        parse_dates={"datetime": ["<DATE>", "<TIME>"]},
        usecols=["<DATE>", "<TIME>", "<CLOSE>"],
        na_values=["nan"],
    ).rename(columns={"<CLOSE>": "Close"})


# Splits


def get_splits(df, cutoff_datetime):
    if isinstance(cutoff_datetime, str):
        cutoff_datetime = datetime.datetime.fromisoformat(cutoff_datetime)
    start_date = df.index.min()
    end_date = df.index.max()
    assert cutoff_datetime > start_date
    assert cutoff_datetime < end_date
    indices = np.arange(len(df))
    return (
        indices[df.index < cutoff_datetime].tolist(),
        indices[df.index >= cutoff_datetime].tolist(),
    )


# Features


def is_us_holiday(dt):
    return dt.strftime("%Y-%m-%d") in holidays.UnitedStates()


def extract_datetime_features(ds):
    df = pd.DataFrame()
    df.index = ds
    df["year"] = ds.year
    df["month"] = ds.month
    df["day"] = ds.day
    df["hour"] = ds.hour
    df["day_of_year"] = ds.day_of_year
    df["week_of_year"] = ds.weekofyear
    df["month_name"] = ds.month_name()
    df["day_name"] = ds.day_name()
    df["is_weekend"] = (ds.day_of_week == 5) | (ds.day_of_week == 6)
    df["is_month_start"] = ds.is_month_start
    df["is_quarter_start"] = ds.is_quarter_start
    df["is_month_end"] = ds.is_month_end
    df["is_year_start"] = ds.is_year_start
    # US holidays
    df["is_holiday"] = pd.Series(ds.values).apply(is_us_holiday).values
    df["is_day_before_holiday"] = (
        pd.Series(ds + datetime.timedelta(days=1)).map(is_us_holiday).values
    )
    df["is_day_after_holiday"] = (
        pd.Series(ds - datetime.timedelta(days=1)).map(is_us_holiday).values
    )
    return df


def add_datetime_features(df):
    return pd.concat([extract_datetime_features(df.index), df], axis=1)


ORDINALS_INFO = []
ORDINALS = [feat for feat, _ in ORDINALS_INFO]

NOMINALS = [
    "hour",
    "month_name",
    "day_name",
    "is_weekend",
    "is_month_start",
    "is_quarter_start",
    "is_month_end",
    "is_year_start",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]

NUMERICALS = ["price"]

UNUSED = []

TARGET_VAR = "price"


def set_col_dtypes(dataf):
    dataf = dataf.drop(columns=UNUSED, errors="ignore")

    for col in NUMERICALS:
        if col not in dataf.columns:
            continue
        dataf[col] = dataf[col].astype("float")

    for col, categories in ORDINALS_INFO:
        if col not in dataf.columns:
            continue
        dataf[col] = dataf[col].astype(
            CategoricalDtype(categories=categories, ordered=True)
        )

    for col in NOMINALS:
        if col not in dataf.columns:
            continue
        dataf[col] = dataf[col].astype("category")

    existing_cols = set(dataf.columns)
    col_order = [
        col for col in NUMERICALS + ORDINALS + NOMINALS if col in existing_cols
    ]
    return dataf[col_order]


def prepare_dataset(df):
    return (
        pd.DataFrame(index=df.index, data=dict(price=df.Close.values))
        .pipe(add_datetime_features)
        .pipe(set_col_dtypes)
    )


# Preprocessing


def get_numerical_cols(dataf):
    return dataf.select_dtypes("number").columns.tolist()


def get_ordinal_cols(dataf):
    return [
        col
        for col in dataf.select_dtypes("category").columns
        if dataf[col].dtypes.ordered
    ]


def get_nominal_cols(dataf):
    return [
        col
        for col in dataf.select_dtypes("category").columns
        if not dataf[col].dtypes.ordered
    ]


def make_preprocessor(x_train: pd.DataFrame):
    from sklearn.pipeline import Pipeline

    numerical_cols = get_numerical_cols(x_train)
    num_transformer = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_cols = sorted(get_ordinal_cols(x_train))
    ordinal_category_list = [
        dt.categories.tolist() for dt in x_train[ordinal_cols].dtypes
    ]
    ordinal_transformer = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_category_list,
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
        ]
    )

    nominal_cols = sorted(get_nominal_cols(x_train))
    nominal_transformer = Pipeline(
        [
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numerical", num_transformer, numerical_cols),
                        ("ordinal", ordinal_transformer, ordinal_cols),
                        ("nominal", nominal_transformer, nominal_cols),
                    ],
                    remainder="drop",
                ),
            )
        ]
    ).fit(x_train)

    if nominal_cols:
        nominal_enc_cols = (
            preprocessor.named_steps["preprocess"]
            .transformers_[2][1]
            .named_steps["encoder"]
            .get_feature_names_out(nominal_cols)
            .tolist()
        )
    else:
        nominal_enc_cols = []

    preprocessor.feature_names_out_ = numerical_cols + ordinal_cols + nominal_enc_cols
    return preprocessor


def make_target_preprocessor(y_train):
    return StandardScaler().fit(y_train.reshape(-1, 1))


# Time-series dataset


def sliding_window(data, window_size: int):
    """Makes snippets of data for sequence prediction by sliding a window with size `look_back`
    Args:
        data (np.array): data with x and y values, shape = (T, F)
        window_size (int): window size
    """
    # shape = (N, W, F)
    return np.array(list(windowed(data, window_size)))


def make_ts_samples(data, look_back, target_idx):
    snippets = sliding_window(data, look_back)
    x = np.swapaxes(snippets[:, :-1, :], 1, 2)  # (N, W-1, F)
    y = snippets[:, -1, target_idx]  # (N, )
    return x, y


def make_ts_dataset_split(train_x, train_y, val_x, val_y):
    x = np.concatenate([train_x, val_x], axis=0)
    y = np.concatenate([train_y, val_y], axis=0)
    splits = list(range(len(train_x))), list(range(len(train_x), len(x)))
    return x, y, splits


# Evaluate
def visualize_predictions(dates, prices, preds):
    prices = prices.reshape(-1, 1)
    preds = preds.reshape(-1, 1)

    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    axes.plot(dates, prices, color="red", label="Real Stock Price")
    axes.plot(dates, preds, color="blue", label="Predicted Stock Price")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"Stock Price")
    plt.legend()

    for metric_name, metric, fmt in [
        ("MSE", mean_squared_error, ".4f"),
        ("R2", r2_score, ".2%"),
        ("MAPE", mean_absolute_percentage_error, ".2%"),
    ]:
        score = metric(prices, preds)
        print(f"{metric_name}: {score:{fmt}}")

    return figure


# Train


def make_arch(architecture):
    if architecture is None:
        return None
    if architecture == "LSTMPlus":
        return LSTMPlus
    if architecture == "InceptionTime":
        return InceptionTime
    if architecture == "InceptionTimePlus":
        return InceptionTimePlus
    raise ValueError(architecture)


def train_eval_infer(
    config,
    df,
    row_splits,
    wandb_run=None,
):
    # preprocessing
    xpp = make_preprocessor(df.iloc[row_splits[0]])
    ypp = make_target_preprocessor(df.iloc[row_splits[0]][TARGET_VAR].values)
    data_pp = xpp.transform(df)

    # split
    target_idx = df.columns.tolist().index(TARGET_VAR)
    look_back = config["data"]["look_back"]  # choose sequence length
    train_x, train_y = make_ts_samples(data_pp[row_splits[0]], look_back, target_idx)
    val_x, val_y = make_ts_samples(data_pp[row_splits[1]], look_back, target_idx)
    x, y, splits = make_ts_dataset_split(train_x, train_y, val_x, val_y)

    # callbacks
    cbs = [SaveModelCallback()]
    early_stop_patience = config["model"].get("early_stop_patience")
    if early_stop_patience:
        cbs.append(EarlyStoppingCallback(patience=early_stop_patience))
    if wandb_run:
        cbs.append(WandbCallback())

    # learn
    bs = config["model"]["batch_size"]
    learn = TSRegressor(
        x,
        y,
        splits=splits,
        bs=bs,
        arch=make_arch(config["model"]["architecture"]),
        metrics=[rmse, mape],
        train_metrics=True,
        cbs=cbs,
    )

    # learning rate
    lr = config["model"]["lr"]
    if lr is None:
        lr_res = learn.lr_find(start_lr=1e-6, end_lr=1e-1, num_it=200)
        lr = lr_res.valley

    # fit
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", category=UndefinedMetricWarning, module=r".*"
        )
        epochs = config["model"]["epochs"]
        learn.fit_one_cycle(epochs, lr)

    learn.remove_cb(SaveModelCallback)
    learn.remove_cb(WandbCallback)
    learn.remove_cb(EarlyStoppingCallback)

    # evaluate

    def inverse_transform_target(y):
        return ypp.inverse_transform(np.array(y).reshape(-1, 1))

    def evaluate(split_idx):
        split_name = ["train", "validation"][split_idx]
        print()
        print(f"{config['model']['architecture']} - {split_name} set")
        print("=" * 80)
        split = splits[split_idx]
        dates = df.iloc[row_splits[split_idx]].index[look_back - 1 :]
        prices = inverse_transform_target(y[split])
        _, _, y_pred = learn.get_X_preds(x[split])
        preds = inverse_transform_target(y_pred)
        fig = visualize_predictions(dates, prices, preds)
        print("=" * 80)
        plt.savefig(FIGURES_DIR / f"{split_name}-backtest.png", dpi=400)

    evaluate(0)
    evaluate(1)
    return xpp, ypp, learn


# Export


def log_file_artifact(wandb_run, path, name, type):
    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(path)
    return wandb_run.log_artifact(artifact)


def log_training_dataset(df, wandb_run=None):
    df = df.reset_index()
    artifact_name = "training_dataframe"

    path = f"{artifact_name}.json"
    df.to_json(path, orient="records")

    if wandb_run:
        log_file_artifact(wandb_run, path, artifact_name, type="dataset")
        wandb.log(
            dict(
                df=wandb.Table(dataframe=df),
            )
        )
    return path


def log_learner(learn, wandb_run=None):
    path = EXPORTS_DIR / "learn.pkl"
    learn.export(path)
    if wandb_run:
        log_file_artifact(wandb_run, path, "learn", type="model")
    return path


def log_preprocessor(pp, name, wandb_run=None):
    path = EXPORTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(pp, f)
    if wandb_run:
        log_file_artifact(wandb_run, path, name, type="preprocessor")
    return path


# Experiment


def run_experiment(config):
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    # wandb
    wandb_run = None
    if config.get("wandb", {}).get("wandb_enabled", False):
        wandb_run = wandb.init(
            project=config["wandb"]["wandb_project"],
            entity=config["wandb"]["wandb_username"],
        )
        config["data"]["features"] = ORDINALS + NOMINALS + NUMERICALS
        wandb.config.update(flatten_dict(config))

    # data
    dataset_path = config["data"]["path"]
    if wandb_run:
        artifact_dir = wandb_run.use_artifact(dataset_path, type="raw_data").download()
        dataset_path = f"./{artifact_dir}/{config['data']['stock_id'].lower()}.us.txt"

    df = load_stock_price_dataset(dataset_path).pipe(prepare_dataset)
    row_splits = get_splits(df, config["data"]["split_date"])
    df["is_validation"] = False
    df.iloc[row_splits[1], df.columns.get_loc("is_validation")] = True
    print("validation/train ratio", len(row_splits[1]) / len(row_splits[0]))

    # experiment
    xpp, ypp, learn = train_eval_infer(
        config,
        df,
        row_splits,
        wandb_run=wandb_run,
    )

    # log artifacts
    log_training_dataset(df, wandb_run)
    log_preprocessor(xpp, "xpp", wandb_run)
    log_preprocessor(ypp, "ypp", wandb_run)
    log_learner(learn, wandb_run)

    # wrap up
    if wandb_run:
        wandb.finish()


def make_experiment_dir(root=".", name=None):
    name = name or generate_time_id()
    experiment_dir = Path(root) / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    args = parser.parse_args()

    with open(args.cfg) as f:
        config = json.load(f)

    with set_dir(make_experiment_dir()):
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        run_experiment(config)
