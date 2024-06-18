# %%

from hazardous.data._seer import (
    load_seer,
    NUMERIC_COLUMN_NAMES,
    CATEGORICAL_COLUMN_NAMES,
)

path_seer = (
    "/Users/vincentmaladiere/dev/hazardous/hazardous/data/"
    "seer_cancer_cardio_raw_data.txt"
)

X, y = load_seer(
    input_path=path_seer,
    survtrace_preprocessing=True,
    return_X_y=True,
)

# %%
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

X, y = make_synthetic_competing_weibull(
    n_events=3,
    n_samples=int(20_000 * 1.58),
    n_features=20,
    censoring_relative_scale=1.5,
    complex_features=False,
    independent_censoring=False,
    return_X_y=True,
)

NUMERIC_COLUMN_NAMES = list(X.columns)
CATEGORICAL_COLUMN_NAMES = []

X.shape, y.shape
# %%


from auton_survival.preprocessing import Preprocessor

X = Preprocessor().fit_transform(
    X,
    cat_feats=CATEGORICAL_COLUMN_NAMES,
    num_feats=NUMERIC_COLUMN_NAMES,
)
X.shape
# %%
from sklearn.model_selection import (
    train_test_split,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y["event"],
    random_state=0,
)
X_train_, X_val, y_train_, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    stratify=y_train["event"],
    random_state=0,
)

duration_train = y_train_["duration"]
event_train = y_train_["event"]

duration_val = y_val["duration"]
event_val = y_val["event"]

X_train_ = X_train_.to_numpy(dtype="float64")
X_val = X_val.to_numpy(dtype="float64")

print(X_train_.shape, X_val.shape)

# %%
from tqdm import tqdm
from auton_survival.models.dsm import DeepSurvivalMachines
from sklearn.model_selection import ParameterGrid

param_grid = {
    "k": [4, 6, 8],
    'distribution' : ['LogNormal', 'Weibull'],
    'learning_rate' : [1e-4, 1e-3],
    'layers' : [[100], [100, 100]]
}
params = ParameterGrid(param_grid)

best_model, best_hp, min_nll = None, None, float("inf")

for param in tqdm(params):
    model = DeepSurvivalMachines(
        k=param['k'],
        distribution=param['distribution'],
        layers=param['layers'],
    )
    model.fit(
        X_train_,
        duration_train,
        event_train,
        iters=100,
        learning_rate=param['learning_rate'],
    )
    nll = model.compute_nll(X_val, duration_val, event_val)
    if nll < min_nll:
        min_nll = nll
        best_model = model
        best_hp = param


# %%
from time import time

tic = time()
duration_train = y_train["duration"]
event_train = y_train["event"]

model = DeepSurvivalMachines(
    k=best_hp['k'],
    distribution=best_hp['distribution'],
    layers=best_hp['layers'],
)
model.fit(
    X_train,
    duration_train,
    event_train,
    iters=100,
    learning_rate=best_hp["learning_rate"],
)
toc = time()
print(f"{toc - tic:.2f}s")

# %%

# Compute ct-index

import numpy as np

def get_y_pred(model, X_test, horizons, n_events=3):
    X_test = X_test.to_numpy(dtype="float64")
    
    time_grid = np.linspace(
        y["duration"].min(),
        y["duration"].max(),
        100,
    )
    time_grid = np.quantile(time_grid, horizons).tolist()

    y_pred = []
    for event_idx in range(1, n_events+1):
        y_pred_event = model.predict_risk(X_test, time_grid, risk=event_idx)
        y_pred.append(y_pred_event[None, :, :])

    return np.asarray(time_grid), np.concatenate(y_pred, axis=0)

# %%

horizons = [0.25, 0.5, 0.75]
time_grid, y_pred = get_y_pred(model, X_test, horizons)
print(y_pred.shape)
y_pred


# %%
from collections import defaultdict
from sksurv.metrics import concordance_index_ipcw


def get_c_index(y_train, y_test, y_pred, time_grid, n_events=3):

    print(y_train["duration"].max())
    print(y_test["duration"].max())
    print(time_grid.max())

    c_indexes = defaultdict(list)

    y_train_binary = y_train.copy()
    y_test_binary = y_test.copy()
    
    for event_idx in range(n_events):

        y_train_binary["event"] = (y_train["event"] == (event_idx + 1)) 
        y_test_binary["event"] = (y_test["event"] == (event_idx + 1))

        et_train = make_recarray(y_train_binary)
        et_test = make_recarray(y_test_binary)

        for time_idx in range(len(time_grid)):
            y_pred_at_t = y_pred[event_idx][:, time_idx]
            tau = time_grid[time_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(                
                et_train,
                et_test,
                y_pred_at_t,
                tau=tau,
            )
            c_indexes[event_idx].append(round(ct_index, 3))
    
    return c_indexes


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


get_c_index(y_train, y_test, y_pred, time_grid)


# %%

from hazardous.metrics._brier_score import integrated_brier_score_incidence

horizons = np.linspace(0, 1, 100)
time_grid, y_pred = get_y_pred(model, X_test, horizons)
print(y_pred.shape)
y_pred

# %%

n_events = 3
all_ibs = []
for event_idx in range(n_events):
    ibs = integrated_brier_score_incidence(
        y_train,
        y_test,
        y_pred[event_idx],
        np.asarray(time_grid),
        event_of_interest=event_idx + 1,
    )
    all_ibs.append(ibs)

print(all_ibs)


# %%


time_grid, y_pred = get_y_pred(model, X_test, horizons)
y_surv = (1 - y_pred.sum(axis=0))[None, :, :]
y_pred = np.concatenate([y_surv, y_pred], axis=0)

print(y_pred.shape)
print(y_pred)


# %%

accuracy_in_time = []

for time_idx in range(len(time_grid)):

    y_pred_time = y_pred[:, :, time_idx]
    mask = (y_test["event"] == 0) & (y_test["duration"] < time_grid[time_idx])
    y_pred_time = y_pred_time[:, ~mask]
    
    y_pred_class = y_pred_time.argmax(axis=0)
    y_test_class = y_test["event"] * (y_test["duration"] < time_grid[time_idx])
    y_test_class = y_test_class.loc[~mask]

    score = (y_test_class.values == y_pred_class).mean()
    accuracy_in_time.append(score)

accuracy_in_time


# %%
