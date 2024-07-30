# %%
from pathlib import Path
from time import time
from tqdm import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from sksurv.metrics import concordance_index_ipcw

from hazardous.data._seer import (
    load_seer,
    NUMERIC_COLUMN_NAMES,
    CATEGORICAL_COLUMN_NAMES,
)
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.utils import make_time_grid
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple

from auton_survival.preprocessing import Preprocessor
from auton_survival.models.dsm import DeepSurvivalMachines

PATH_SCORES = Path("../benchmark/scores")
PATH_SEER = Path("../hazardous/data/seer_cancer_cardio_raw_data.txt")
WEIBULL_PARAMS = {
    "n_events": 3,
    "n_samples": 20_000,
    "censoring_relative_scale": 1.5,
    "complex_features": False,
    "independent_censoring": False,
}
SEEDS = range(5)
N_STEPS_TIME_GRID = 20
MODEL_NAME = "DSM"
N_ITERS = 100


def run_evaluation(dataset_name):
    all_scores = []

    for random_state in tqdm(SEEDS):
        scores = run_seed(dataset_name, random_state)
        all_scores.append(scores)
        
        path_dir = PATH_SCORES / "raw" / MODEL_NAME
        path_dir.mkdir(parents=True, exist_ok=True)
        path_raw_scores = path_dir / f"{dataset_name}.json"
        json.dump(all_scores, open(path_raw_scores, "w"))


def run_seed(dataset_name, random_state):
    bunch, dataset_params = get_dataset(dataset_name, random_state)
    best_hp = get_best_hp(dataset_name, random_state, bunch)
    model, fit_time = get_model(bunch, best_hp)
    
    scores = evaluate(
        model,
        bunch,
        dataset_name,
        dataset_params=dataset_params,
        model_name=MODEL_NAME,
    )
    scores["fit_time"] = fit_time

    return scores


def evaluate(
    model, bunch, dataset_name, dataset_params, model_name, verbose=True
):
    """Evaluate a model against its test set.
    """
    X_train, y_train = bunch["X_train"], bunch["y_train"]
    X_test, y_test = bunch["X_test"], bunch["y_test"]

    n_events = np.unique(y_train["event"]).shape[0] - 1
    is_competing_risk = n_events > 1

    scores = {
        "is_competing_risk": is_competing_risk,
        "n_events": n_events,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_rows": X_train.shape[0],
        "n_cols": X_train.shape[1],
        "censoring_rate": (y_train["event"] == 0).mean(),
        **dataset_params,
    }

    y_pred, predict_time = get_y_pred(model, bunch)
    time_grid = np.asarray(bunch["time_grid"])

    print(f"{time_grid=}")
    print(f"{y_pred.shape=}")
    print(f"{y_pred.mean(axis=1).mean(axis=1)=}")

    scores["time_grid"] = time_grid.round(4).tolist()
    scores["y_pred"] = y_pred.round(4).tolist()
    scores["predict_time"] = round(predict_time, 2)

    event_specific_ibs, event_specific_brier_scores = [], []
    event_specific_c_index = []

    if verbose:
        print("Computing Brier scores, IBS and C-index")

    for event_id in range(1, n_events+1):

        # Brier score and IBS
        if dataset_name == "weibull":
            # Use oracle metrics with the synthetic dataset.
            ibs = integrated_brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring.loc[y_test.index],
                scale_censoring=bunch.scale_censoring.loc[y_test.index],
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence_oracle(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                shape_censoring=bunch.shape_censoring.loc[y_test.index],
                scale_censoring=bunch.scale_censoring.loc[y_test.index],
                event_of_interest=event_id,  
            )
        else:
            ibs = integrated_brier_score_incidence(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                event_of_interest=event_id,
            )
            brier_scores = brier_score_incidence(
                y_train,
                y_test,
                y_pred[event_id],
                time_grid,
                event_of_interest=event_id,
            )   
            
        event_specific_ibs.append({
            "event": event_id,
            "ibs": round(ibs, 4),
        })
        event_specific_brier_scores.append({
            "event": event_id,
            "time": list(time_grid.round(2)),
            "brier_score": list(brier_scores.round(4)),
        })

        # C-index
        y_train_binary = y_train.copy()
        y_test_binary = y_test.copy()

        y_train_binary["event"] = (y_train["event"] == event_id)
        y_test_binary["event"] = (y_test["event"] == event_id)

        truncation_quantiles = [0.25, 0.5, 0.75]
        taus = np.quantile(time_grid, truncation_quantiles)
        if event_id == 1:
            print(f"{taus=}")
        taus = tqdm(
            taus,
            desc=f"c-index at tau for event {event_id}",
            total=len(taus),
        )
        c_indices = []
        for tau in taus:
            tau_idx = np.searchsorted(time_grid, tau)
            y_pred_at_t = y_pred[event_id][:, tau_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                make_recarray(y_train_binary),
                make_recarray(y_test_binary),
                y_pred_at_t,
                tau=tau,
            )
            c_indices.append(round(ct_index, 4))

        event_specific_c_index.append({
            "event": event_id,
            "time_quantile": truncation_quantiles,
            "c_index": c_indices,
        })

    scores.update({
        "event_specific_ibs": event_specific_ibs,
        "event_specific_brier_scores": event_specific_brier_scores,
        "event_specific_c_index": event_specific_c_index,
    })

    if is_competing_risk:
        # Accuracy in time

        truncation_quantiles = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        taus = np.quantile(time_grid, truncation_quantiles)
        accuracy = []

        print("Computing accuracy in time")
        print(f"{taus=}")
        
         # TODO: put it into a function in hazardous._metrics
        for tau in taus:
            tau_idx = np.searchsorted(time_grid, tau)
            y_pred_at_t = y_pred[:, :, tau_idx]
            mask = (y_test["event"] == 0) & (y_test["duration"] < tau)
            y_pred_class = y_pred_at_t[:, ~mask].argmax(axis=0)
            y_test_class = y_test["event"] * (y_test["duration"] < tau)
            y_test_class = y_test_class.loc[~mask]
            accuracy.append(
                round(
                    (y_test_class.values == y_pred_class).mean(),
                    4
                )
            )
        scores["accuracy_in_time"] = {
            "time_quantile": truncation_quantiles,
            "accuracy": accuracy,
        }

    else:
        # Yana loss
        if verbose:
            print("Computing Censlog")

        censlog = CensoredNegativeLogLikelihoodSimple().loss(
            y_pred, y_test["duration_test"], y_test["event"], time_grid
        )
        scores["censlog"] = round(censlog, 4)        

    print(f"{event_specific_ibs=}")
    print(f"{event_specific_c_index}")
    print(f"{accuracy=}")

    return scores


def get_dataset(dataset_name, random_state):
    bunch, dataset_params = load_dataset(dataset_name, random_state)
    X, y = bunch["X"], bunch["y"]
    print(f"{X.shape=}, {y.shape=}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=random_state,
    )
    X_train_, X_val, y_train_, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        stratify=y_train["event"],
        random_state=random_state,
    )

    event_train = y_train_["event"]

    print(f"{X_train_.shape=}, {X_val.shape=}")

    pre = Preprocessor()
    pre.fit(
        X_train_,
        cat_feats=bunch["categorical_columns"],
        num_feats=bunch["numeric_columns"],
    )
    X_train_ = pre.transform(X_train_)
    X_val = pre.transform(X_val)

    X_train_ = X_train_.to_numpy(dtype="float64")
    X_val = X_val.to_numpy(dtype="float64")

    pre = Preprocessor()
    pre.fit(
        X_train,
        cat_feats=bunch["categorical_columns"],
        num_feats=bunch["numeric_columns"],
    )
    X_train = pre.transform(X_train)
    X_test = pre.transform(X_test)
    print(f"{X_train.shape=}, {X_test.shape=}")

    X_train = X_train.to_numpy(dtype="float64")
    X_test = X_test.to_numpy(dtype="float64")
    
    n_features = X_train.shape[1]
    
    n_events = len(set(np.unique(event_train)) - {0})
    time_grid = make_time_grid(
        y_test["duration"], n_steps=N_STEPS_TIME_GRID
    ).tolist()
    

    bunch.update({
        "X_train_": X_train_,
        "y_train_": y_train_,
        "X_val": X_val,
        "y_val": y_val,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_features": n_features,
        "n_events": n_events,
        "time_grid": time_grid,
    })
    
    return bunch, dataset_params


def load_dataset(dataset_name, random_state):

    dataset_params = {"random_state": random_state}

    if dataset_name == "seer":
        X, y = load_seer(
            input_path=PATH_SEER,
            survtrace_preprocessing=True,
            return_X_y=True,
        )
        X = X.dropna()
        y = y.iloc[X.index]
        bunch = {
            "X": X,
            "y": y,
            "numeric_columns": NUMERIC_COLUMN_NAMES,
            "categorical_columns": CATEGORICAL_COLUMN_NAMES,
        }

    elif dataset_name == "weibull":
        dataset_params.update(WEIBULL_PARAMS)
        bunch = make_synthetic_competing_weibull(**dataset_params)
        bunch.update({
            "numeric_columns": list(bunch.X.columns),
            "categorical_columns": [],
        })
    else:
        raise ValueError(dataset_name)

    return bunch, dataset_params


def get_model(bunch, best_hp):

    y_train = bunch["y_train"]
    duration_train = y_train["duration"]
    event_train = y_train["event"]

    tic = time()
    model = DeepSurvivalMachines(
        k=best_hp['k'],
        distribution=best_hp['distribution'],
        layers=best_hp['layers'],
    )
    model.fit(
        bunch["X_train"],
        duration_train,
        event_train,
        iters=N_ITERS,
        learning_rate=best_hp["learning_rate"],
    )
    fit_time = time() - tic

    return model, fit_time


def get_best_hp(dataset_name, random_state, bunch):
    path_dir = Path("best_hp") / dataset_name
    path_dir.mkdir(parents=True, exist_ok=True) 
    path_file = path_dir / f"random_state{random_state}.json"

    if path_file.exists():
        best_hp = json.load(open(path_file))
    else:
        best_hp = hp_search(bunch)
        json.dump(best_hp, open(path_file, "w"))
        print(f"Wrote {path_file}")

    return best_hp


def hp_search(bunch):
    param_grid = {
        "k": [4, 6, 8],
        'distribution' : ['LogNormal'],
        'learning_rate' : [1e-4, 1e-3],
        'layers' : [[100, 100]]
    }
    params = ParameterGrid(param_grid)

    best_hp, min_nll = None, float("inf")

    y_train = bunch["y_train_"]
    event_train, duration_train = y_train["event"], y_train["duration"]
    
    y_val = bunch["y_val"]
    event_val, duration_val = y_val["event"], y_val["duration"]

    for param in tqdm(params):
        print(f"{param=}")
        model = DeepSurvivalMachines(
            k=param['k'],
            distribution=param['distribution'],
            layers=param['layers'],
        )
        model.fit(
            bunch["X_train_"],
            duration_train,
            event_train,
            iters=N_ITERS,
            learning_rate=param['learning_rate'],
        )
        nll = model.compute_nll(bunch["X_val"], duration_val, event_val)
        print(f"{nll=}")
        if nll < min_nll:
            print("new best!")
            min_nll = nll
            best_hp = param

    print(f"{best_hp=}")

    return best_hp


def get_y_pred(model, bunch):
    tic = time()
    y_pred = []
    for event_idx in range(1, bunch["n_events"] + 1):
        y_pred_event = model.predict_risk(
            bunch["X_test"],
            bunch["time_grid"],
            risk=event_idx
        )
        y_pred.append(y_pred_event[None, :, :])

    y_pred = np.concatenate(y_pred, axis=0)
    y_surv = (1 - y_pred.sum(axis=0))[None, :, :]
    y_pred = np.concatenate([y_surv, y_pred], axis=0)
    predict_time = time() - tic

    return y_pred, predict_time


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )

# %%
if __name__ == "__main__":
    run_evaluation("seer")

# %%
