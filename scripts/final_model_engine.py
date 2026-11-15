#!/usr/bin/env python3
"""Frozen, donor-selected residual forecasting engine for the final AQI study.

Public API: ``predictions, fit_log = train_all(bundle, outdir, config)``.
The canonical frame has station_id, city, date (label date), issue_date,
horizon (1/2/3), y, last, and x0..x6 (oldest to newest daily AQI).
Each selected city has one representative station and complete horizon triples.

Target ``validation`` is an evaluation-only 2024 period. It is never supplied to
any fitting or selection function. All choices use held-out donor cities.
Outputs are continuous, unbounded AQI. Source scaling is fixed, not fitted.
Run ``python final_model_engine.py --smoke --out /tmp/aqi-engine-smoke`` to test
all architectures, first-order meta updates, resume, and prediction alignment
on artificial data (no project outcome data).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F


ENGINE_VERSION = "final-residual-v1.1"
ARCHITECTURES = ("gru", "lstm", "tcn", "transformer", "mlp")
BASELINES = ("persistence", "seasonal7", "mean7", "ridge_local", "histgb_local", "randomforest_local")
METHODS = (*BASELINES, *(f"{a}_{m}" for a in ARCHITECTURES for m in ("local", "pooled_zero", "finetune")),
           "reptile_gru", "fomaml_gru")
REQUIRED = ("station_id", "city", "date", "issue_date", "horizon", "y", "last", *(f"x{i}" for i in range(7)))
PREDICTION_COLUMNS = ("station_id", "city", "date", "issue_date", "horizon", "budget", "seed",
                      "method", "y_true", "y_pred", "last", "period")
DEFAULTS: dict[str, Any] = {
    "seeds": [17, 43, 71], "budgets": [30, 90, 180],
    "architectures": list(ARCHITECTURES), "hidden_size": 32,
    "batch_size": 128, "num_threads": 2, "device": "auto",
    "pooled_steps": 600, "pooled_lr": 0.001, "eval_every": 60,
    "weight_decay": 0.0001, "grad_clip": 1.0,
    # Scratch and fine-tuning get independent choices from equally sized grids.
    "adaptation_presets": [{"lr": 0.001, "steps": 100}, {"lr": 0.0003, "steps": 100}],
    "scratch_presets": [{"lr": 0.001, "steps": 100}, {"lr": 0.003, "steps": 250}],
    "pseudo_budgets": None, "pseudo_query_days": None,
    "meta_steps": 360, "meta_tasks": 3, "meta_inner_steps": 3,
    "meta_inner_lr": 0.01, "meta_outer_lr": 0.001,
    "reptile_outer_lr": 0.1, "meta_query_days": 14,
    "meta_validation_budget": 90, "purge_days": 9,
    "dev_cities": ["Chandigarh", "Dewas", "Jalandhar", "Ajmer"],
    "ridge_alpha": 10.0, "histgb_iterations": 100, "rf_trees": 128,
    "ridge_presets": [{"alpha": 1.0}, {"alpha": 100.0}],
    "histgb_presets": [{"min_samples_leaf": 5, "l2_regularization": 1.0},
                        {"min_samples_leaf": 15, "l2_regularization": 10.0}],
    "randomforest_presets": [{"min_samples_leaf": 1, "max_depth": None},
                              {"min_samples_leaf": 3, "max_depth": 6}],
    "resume": True, "save_target_states": True, "smoke": False,
}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (Path, pd.Timestamp)):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip" if path.name.endswith(".gz") else None)
    os.replace(temp, path)


def _torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _torch_load(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    # These are locally generated, trusted experiment checkpoints, never input data.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument.
        return torch.load(path, map_location=device)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_") + "_" + hashlib.sha256(s.encode()).hexdigest()[:8]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    c = copy.deepcopy(DEFAULTS)
    c.update(config or {})
    for key in ("seeds", "budgets"):
        c[key] = [int(v) for v in c[key]]
        if not c[key] or len(set(c[key])) != len(c[key]):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    if any(b < 1 for b in c["budgets"]):
        raise ValueError("Budgets must be positive")
    if not set(c["architectures"]).issubset(ARCHITECTURES):
        raise ValueError("Unknown neural architecture")
    if int(c["purge_days"]) < 9:
        raise ValueError("Meta/pseudo-task purge must have at least nine intervening calendar days")
    for grid in ("adaptation_presets", "scratch_presets"):
        if len(c[grid]) != 2 and not c["smoke"]:
            raise ValueError(f"{grid} must contain exactly two donor-tested presets")
        if not c[grid] or any(float(p["lr"]) <= 0 or int(p["steps"]) < 1 for p in c[grid]):
            raise ValueError(f"Invalid {grid}")
    if len(c["adaptation_presets"]) != len(c["scratch_presets"]):
        raise ValueError("Scratch and fine-tuning must have equally sized tuning grids")
    for k in ("batch_size", "eval_every", "num_threads", "meta_tasks", "meta_inner_steps", "meta_query_days"):
        if int(c[k]) < 1:
            raise ValueError(f"{k} must be positive")
    return c


def canonical_frame(frame: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """Validate, copy and sort without silently dropping or imputing a row."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    d = frame.copy()
    if len(d) == 0:
        raise ValueError(f"{name}: empty frame")
    for col in ("station_id", "city"):
        if d[col].isna().any():
            raise ValueError(f"{name}: null {col}")
        d[col] = d[col].astype(str)
    for col in ("date", "issue_date"):
        d[col] = pd.to_datetime(d[col], errors="raise").dt.normalize()
    for col in ("horizon", "y", "last", *(f"x{i}" for i in range(7))):
        d[col] = pd.to_numeric(d[col], errors="raise")
        if not np.isfinite(d[col].to_numpy(float)).all():
            raise ValueError(f"{name}: nonfinite {col}")
    if not d.horizon.isin([1, 2, 3]).all():
        raise ValueError(f"{name}: horizons must be 1/2/3")
    d["horizon"] = d.horizon.astype(int)
    if not ((d.date - d.issue_date).dt.days == d.horizon).all():
        raise ValueError(f"{name}: label date must equal issue date plus horizon")
    if not np.allclose(d["last"], d.x6, rtol=0, atol=1e-6):
        raise ValueError(f"{name}: last must equal newest observed lag x6")
    if d.duplicated(["city", "date", "horizon"]).any():
        raise ValueError(f"{name}: duplicate city/date/horizon; exactly one representative station is required")
    if (d.groupby("city").station_id.nunique() != 1).any():
        raise ValueError(f"{name}: more than one station per city")
    triples = d.groupby(["city", "date"]).horizon.agg(["count", "nunique"])
    if not ((triples["count"] == 3) & (triples["nunique"] == 3)).all():
        raise ValueError(f"{name}: every label date must contain all three horizons")
    for col, expected in (("feature_date_min", d.issue_date - pd.Timedelta(days=6)),
                          ("feature_date_max", d.issue_date)):
        if col in d:
            actual = pd.to_datetime(d[col], errors="raise").dt.normalize()
            if not (actual == expected).all():
                raise ValueError(f"{name}: {col} does not match the seven-day past-only input")
            d[col] = actual
    if "issue_doy" in d and not (pd.to_numeric(d.issue_doy) == d.issue_date.dt.day_of_year).all():
        raise ValueError(f"{name}: inconsistent issue_doy")
    if "issue_dow" in d and not (pd.to_numeric(d.issue_dow) == d.issue_date.dt.day_of_week).all():
        raise ValueError(f"{name}: inconsistent issue_dow (Monday must be zero)")
    return d.sort_values(["city", "date", "horizon"]).reset_index(drop=True)


def select_budget(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Last B common label dates, exactly 3B rows; caller supplies one city."""
    if frame.city.nunique() != 1:
        raise ValueError("select_budget requires exactly one city")
    dates = np.sort(frame.date.unique())
    if len(dates) < budget:
        raise ValueError(f"Only {len(dates)} label dates available for budget {budget}")
    d = frame[frame.date.isin(dates[-budget:])].copy().reset_index(drop=True)
    if len(d) != 3 * budget or d.date.nunique() != budget:
        raise AssertionError("Budget must be exactly B labels and 3B examples")
    return d


def feature_array(frame: pd.DataFrame) -> np.ndarray:
    """7x6 fixed-scale features. Calendar phases use the forecast issue day."""
    n = len(frame)
    x = np.empty((n, 7, 6), dtype=np.float32)
    x[:, :, 0] = frame[[f"x{i}" for i in range(7)]].to_numpy(np.float32) / 500.0
    doy = frame.issue_date.dt.day_of_year.to_numpy(float)
    dow = frame.issue_date.dt.day_of_week.to_numpy(float)
    calendar = np.column_stack((np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                                np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
                                frame.horizon.to_numpy(float) / 3.0))
    x[:, :, 1:] = calendar[:, None, :]
    return x


class ResidualForecaster(nn.Module):
    def __init__(self, architecture: str, hidden: int = 32):
        super().__init__()
        self.architecture = architecture
        self.hidden = hidden
        if architecture == "gru":
            self.body = nn.GRU(6, hidden, batch_first=True)
        elif architecture == "lstm":
            self.body = nn.LSTM(6, hidden, batch_first=True)
        elif architecture == "tcn":
            self.conv1 = nn.Conv1d(6, hidden, 3)
            self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=2)
        elif architecture == "transformer":
            if hidden % 4:
                raise ValueError("Transformer hidden size must be divisible by four")
            self.embed = nn.Linear(6, hidden)
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.0, activation="gelu",
                                                batch_first=True, norm_first=False)
            self.body = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            positions = torch.arange(7).float()[:, None]
            div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
            pos = torch.zeros(7, hidden)
            pos[:, 0::2] = torch.sin(positions * div)
            pos[:, 1::2] = torch.cos(positions * div)
            self.register_buffer("positions", pos[None, :, :])
        elif architecture == "mlp":
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(42, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        else:
            raise ValueError(f"Unknown architecture {architecture}")
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture in ("gru", "lstm"):
            z = self.body(x)[0][:, -1]
        elif self.architecture == "tcn":
            z = F.relu(self.conv1(F.pad(x.transpose(1, 2), (2, 0))))
            z = F.relu(self.conv2(F.pad(z, (4, 0))))[:, :, -1]
        elif self.architecture == "transformer":
            z = self.body(self.embed(x) + self.positions)[:, -1]
        else:
            z = self.body(x)
        return self.head(z).squeeze(-1)


def _new_model(architecture: str, seed: int, c: dict[str, Any], device: torch.device) -> ResidualForecaster:
    _seed(seed)
    return ResidualForecaster(architecture, int(c["hidden_size"])).to(device)


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@dataclass
class ArrayData:
    x: torch.Tensor
    residual: torch.Tensor
    cities: np.ndarray
    dates: np.ndarray


def _arrays(frame: pd.DataFrame, device: torch.device) -> ArrayData:
    return ArrayData(torch.as_tensor(feature_array(frame), device=device),
                     torch.as_tensor((frame.y.to_numpy(np.float32) - frame["last"].to_numpy(np.float32)) / 100.0,
                                     device=device),
                     frame.city.to_numpy(str), frame.date.to_numpy())


class CitySampler:
    """Uniform city, then uniform example within city, using a local RNG."""
    def __init__(self, cities: np.ndarray, seed: int):
        self.groups = [np.flatnonzero(cities == city) for city in sorted(set(cities))]
        self.rng = np.random.default_rng(seed)
        self.n = len(cities)

    def sample(self, batch_size: int) -> np.ndarray:
        groups = self.rng.integers(len(self.groups), size=min(batch_size, self.n))
        result = np.empty(len(groups), dtype=np.int64)
        for k in np.unique(groups):
            pos = np.flatnonzero(groups == k)
            result[pos] = self.rng.choice(self.groups[int(k)], size=len(pos), replace=True)
        return result


def predict_neural(model: nn.Module, frame: pd.DataFrame, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    x = feature_array(frame)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            parts.append(model(torch.as_tensor(x[start:start + batch_size], device=device)).cpu().numpy())
    residual = np.concatenate(parts).astype(float)
    result = frame["last"].to_numpy(float) + residual * 100.0
    if not np.isfinite(result).all():
        raise FloatingPointError("Neural predictions are nonfinite")
    return result


def _macro_mae(model: nn.Module, frame: pd.DataFrame, device: torch.device) -> float:
    errors = frame[["city", "horizon"]].copy()
    errors["error"] = np.abs(predict_neural(model, frame, device) - frame.y.to_numpy(float))
    return float(errors.groupby(["city", "horizon"]).error.mean().mean())


@dataclass
class FitResult:
    model: ResidualForecaster
    selected_step: int
    validation_mae: float | None
    history: list[dict[str, Any]]
    elapsed: float
    episodes: list[dict[str, Any]] | None = None


def fit_neural(model: ResidualForecaster, train: pd.DataFrame, *, steps: int, lr: float, seed: int,
               c: dict[str, Any], device: torch.device, validation: pd.DataFrame | None = None,
               checkpoint: Path | None = None, select_best: bool = False) -> FitResult:
    """City-balanced Adam/MSE; checkpoints selected only by uncapped donor MAE."""
    start = time.perf_counter()
    data = _arrays(train, device)
    sampler = CitySampler(data.cities, seed + 991)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(c["weight_decay"]))
    history: list[dict[str, Any]] = []
    first_step, elapsed_before = 0, 0.0
    best_mae = float("inf")
    best_state, best_step = _state(model), 0
    if checkpoint is not None and checkpoint.exists() and c["resume"]:
        saved = _torch_load(checkpoint, device)
        if int(saved["requested_steps"]) != steps or float(saved["lr"]) != float(lr):
            raise ValueError(f"Checkpoint training settings differ: {checkpoint}")
        model.load_state_dict(saved["state_dict"])
        opt.load_state_dict(saved["optimizer"])
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        first_step = int(saved["step"])
        history = saved["history"]
        best_state, best_step, best_mae = saved["best_state"], int(saved["best_step"]), float(saved["best_mae"])
        elapsed_before = float(saved["elapsed"])
    elif validation is not None:
        best_mae = _macro_mae(model, validation, device)
        history.append({"step": 0, "loss": None, "validation_mae": best_mae})
    if select_best and validation is None:
        raise ValueError("Best-checkpoint selection requires a donor validation frame")
    for step in range(first_step + 1, int(steps) + 1):
        idx = torch.as_tensor(sampler.sample(int(c["batch_size"])), device=device)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(data.x[idx]), data.residual[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
        opt.step()
        if step % int(c["eval_every"]) == 0 or step == steps:
            mae = _macro_mae(model, validation, device) if validation is not None else None
            history.append({"step": step, "loss": float(loss.detach().cpu()), "validation_mae": mae})
            if mae is not None and mae < best_mae - 1e-9:
                best_mae, best_step, best_state = mae, step, _state(model)
            if checkpoint is not None:
                _torch_save(checkpoint, {"state_dict": _state(model), "optimizer": opt.state_dict(),
                    "step": step, "requested_steps": steps, "lr": lr, "sampler_rng": sampler.rng.bit_generator.state,
                    "history": history, "best_state": best_state, "best_step": best_step, "best_mae": best_mae,
                    "elapsed": elapsed_before + time.perf_counter() - start})
    elapsed = elapsed_before + time.perf_counter() - start
    if select_best:
        model.load_state_dict(best_state)
    else:
        best_step = int(steps)
    return FitResult(model, best_step, best_mae if validation is not None else None, history, elapsed)


def _pseudo_tasks(train: pd.DataFrame, query: pd.DataFrame, cities: list[str], budgets: list[int],
                  c: dict[str, Any]) -> list[tuple[str, int, pd.DataFrame, pd.DataFrame]]:
    tasks = []
    for city in cities:
        q = query[query.city == city].copy()
        if q.empty:
            raise ValueError(f"No donor pseudo-query for {city}")
        if c["pseudo_query_days"] is not None:
            dates = np.sort(q.date.unique())[:int(c["pseudo_query_days"])]
            q = q[q.date.isin(dates)].copy()
        # Nine *intervening* days is stricter than just a nine-day date difference.
        latest_support = q.date.min() - pd.Timedelta(days=int(c["purge_days"]) + 1)
        support = train[(train.city == city) & (train.date <= latest_support)]
        for budget in budgets:
            s = select_budget(support, int(budget))
            if (q.date.min() - s.date.max()).days <= int(c["purge_days"]):
                raise AssertionError("Pseudo-task temporal purge violated")
            tasks.append((city, int(budget), s, q))
    return tasks


def _adapt(model: ResidualForecaster, support: pd.DataFrame, preset: dict[str, Any], seed: int,
           c: dict[str, Any], device: torch.device) -> FitResult:
    return fit_neural(model, support, steps=int(preset["steps"]), lr=float(preset["lr"]), seed=seed,
                      c=c, device=device)


def tune_presets(architecture: str, base: ResidualForecaster | None,
                 tasks: list[tuple[str, int, pd.DataFrame, pd.DataFrame]], presets: list[dict[str, Any]],
                 *, seed: int, c: dict[str, Any], device: torch.device, label: str,
                 out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = out / "selection" / f"{label}.json"
    if cache.exists() and c["resume"]:
        saved = json.loads(cache.read_text())
        return saved["selected_preset"], saved["scores"]
    scores = []
    for candidate, preset in enumerate(presets):
        for task_idx, (city, budget, support, query) in enumerate(tasks):
            model = _new_model(architecture, seed, c, device) if base is None else copy.deepcopy(base).to(device)
            result = _adapt(model, support, preset, seed + task_idx * 37, c, device)
            scores.append({"selection": label, "candidate": candidate, "lr": float(preset["lr"]),
                "steps": int(preset["steps"]), "city": city, "budget": budget,
                "macro_horizon_mae": _macro_mae(result.model, query, device), "seed": seed,
                "support_first": support.date.min(), "support_last": support.date.max(),
                "query_first": query.date.min(), "query_last": query.date.max(),
                "purge_intervening_days": (query.date.min() - support.date.max()).days - 1,
                "fit_seconds": result.elapsed, "support_examples": len(support),
                "query_examples": len(query), "architecture": architecture,
                "parameters": sum(p.numel() for p in result.model.parameters())})
            del model, result
    table = pd.DataFrame(scores)
    # Equal weight for each held-out city, label budget and horizon; strict tie-break by grid order.
    ranking = table.groupby("candidate", sort=True).macro_horizon_mae.mean()
    chosen = int(ranking.idxmin())
    selected = copy.deepcopy(presets[chosen])
    _atomic_json(cache, {"selected_candidate": chosen, "selected_preset": selected,
        "mean_macro_mae": ranking.to_dict(), "scores": scores,
        "selection_source": "held-out donor-city 2023 pseudo-query; targets entirely excluded"})
    return selected, scores


class EpisodeSampler:
    """Uniform donor-city tasks with disjoint, chronological support/query labels."""
    def __init__(self, train: pd.DataFrame, budgets: list[int], query_days: int, purge_days: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.groups: dict[str, dict[str, Any]] = {}
        self.budgets, self.query_days, self.purge_days = budgets, query_days, purge_days
        for city in sorted(train.city.unique()):
            g = train[train.city == city]
            dates = np.sort(g.date.unique())
            index = {date: g.index[g.date == date].to_numpy() for date in dates}
            # Require the largest budget for every sampled city; no city is silently underweighted.
            starts = []
            for j in range(len(dates) - query_days + 1):
                cutoff = dates[j] - np.timedelta64(purge_days + 1, "D")
                n_support = int(np.searchsorted(dates, cutoff, side="right"))
                if n_support >= max(budgets):
                    starts.append((j, n_support))
            if not starts:
                raise ValueError(f"Donor {city} cannot form a purged {max(budgets)}-label meta episode")
            self.groups[str(city)] = {"dates": dates, "index": index, "starts": starts}
        self.cities = sorted(self.groups)

    def sample(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        city = self.cities[int(self.rng.integers(len(self.cities)))]
        g = self.groups[city]
        budget = int(self.rng.choice(self.budgets))
        j, count = g["starts"][int(self.rng.integers(len(g["starts"])))]
        sd = g["dates"][count - budget:count]
        qd = g["dates"][j:j + self.query_days]
        support = np.concatenate([g["index"][date] for date in sd])
        query = np.concatenate([g["index"][date] for date in qd])
        gap = int((qd[0] - sd[-1]) / np.timedelta64(1, "D")) - 1
        if gap < self.purge_days or np.intersect1d(sd, qd).size:
            raise AssertionError("Meta support/query leakage")
        return support, query, {"city": city, "budget": budget, "support_first": str(sd[0])[:10],
            "support_last": str(sd[-1])[:10], "query_first": str(qd[0])[:10], "query_last": str(qd[-1])[:10],
            "purge_intervening_days": gap, "support_examples": len(support), "query_examples": len(query)}


def _inner_updates(model: ResidualForecaster, data: ArrayData, indices: np.ndarray, rng: np.random.Generator,
                   steps: int, lr: float, c: dict[str, Any], device: torch.device) -> None:
    # No differentiable optimizer is used: query gradients for FOMAML are explicitly first order.
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        selected = rng.choice(indices, size=min(int(c["batch_size"]), len(indices)), replace=True)
        idx = torch.as_tensor(selected, device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(data.x[idx]), data.residual[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite meta inner loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
        opt.step()


def _meta_validation(model: ResidualForecaster, tasks: list[tuple[str, int, pd.DataFrame, pd.DataFrame]],
                      c: dict[str, Any], device: torch.device, seed: int) -> float:
    errors = []
    for i, (_, _, support, query) in enumerate(tasks):
        adapted = copy.deepcopy(model).to(device)
        data = _arrays(support, device)
        _inner_updates(adapted, data, np.arange(len(support)), np.random.default_rng(seed + i),
                       int(c["meta_inner_steps"]), float(c["meta_inner_lr"]), c, device)
        errors.append(_macro_mae(adapted, query, device))
        del adapted, data
    return float(np.mean(errors))


def fit_meta(algorithm: str, train: pd.DataFrame, *, steps: int, seed: int, c: dict[str, Any],
             device: torch.device, validation_tasks: list | None = None,
             checkpoint: Path | None = None, select_best: bool = False) -> FitResult:
    if algorithm not in ("reptile", "fomaml"):
        raise ValueError(algorithm)
    start = time.perf_counter()
    model = _new_model("gru", seed, c, device)
    train = train.reset_index(drop=True)
    data = _arrays(train, device)
    sampler = EpisodeSampler(train, list(c["budgets"]), int(c["meta_query_days"]), int(c["purge_days"]), seed + 177)
    rng = np.random.default_rng(seed + 7919)
    outer = torch.optim.Adam(model.parameters(), lr=float(c["meta_outer_lr"])) if algorithm == "fomaml" else None
    history, episodes = [], []
    best_state, best_step, best_mae = _state(model), 0, float("inf")
    first_step, elapsed_before = 0, 0.0
    if checkpoint is not None and checkpoint.exists() and c["resume"]:
        saved = _torch_load(checkpoint, device)
        if saved["algorithm"] != algorithm or int(saved["requested_steps"]) != steps:
            raise ValueError("Meta checkpoint settings differ")
        model.load_state_dict(saved["state_dict"])
        if outer is not None:
            outer.load_state_dict(saved["optimizer"])
        sampler.rng.bit_generator.state, rng.bit_generator.state = saved["sampler_rng"], saved["inner_rng"]
        first_step, history, episodes = int(saved["step"]), saved["history"], saved["episodes"]
        best_state, best_step, best_mae = saved["best_state"], int(saved["best_step"]), float(saved["best_mae"])
        elapsed_before = float(saved["elapsed"])
    elif validation_tasks is not None:
        best_mae = _meta_validation(model, validation_tasks, c, device, seed)
        history.append({"step": 0, "query_mse": None, "validation_adapted_mae": best_mae})
    if select_best and validation_tasks is None:
        raise ValueError("Meta checkpoint selection needs donor validation tasks")
    for step in range(first_step + 1, steps + 1):
        accum = [torch.zeros_like(p) for p in model.parameters()]
        query_losses = []
        for task_number in range(int(c["meta_tasks"])):
            si, qi, audit = sampler.sample()
            audit.update({"step": step, "task": task_number, "algorithm": algorithm, "seed": seed})
            episodes.append(audit)
            adapted = copy.deepcopy(model).to(device)
            _inner_updates(adapted, data, si, rng, int(c["meta_inner_steps"]), float(c["meta_inner_lr"]), c, device)
            query_idx = torch.as_tensor(qi, device=device)
            if algorithm == "fomaml":
                adapted.zero_grad(set_to_none=True)
                query_loss = F.mse_loss(adapted(data.x[query_idx]), data.residual[query_idx])
                if not torch.isfinite(query_loss):
                    raise FloatingPointError("Nonfinite FOMAML query loss")
                query_loss.backward()
                for total, p in zip(accum, adapted.parameters()):
                    if p.grad is not None:
                        total.add_(p.grad.detach())
            else:
                with torch.no_grad():
                    query_loss = F.mse_loss(adapted(data.x[query_idx]), data.residual[query_idx])
                    if not torch.isfinite(query_loss):
                        raise FloatingPointError("Nonfinite Reptile query loss")
                    for total, p in zip(accum, adapted.parameters()):
                        total.add_(p.detach())
            query_losses.append(float(query_loss.detach().cpu()))
            del adapted
        tasks = int(c["meta_tasks"])
        if algorithm == "fomaml":
            outer.zero_grad(set_to_none=True)
            # Copy gradients at adapted parameters to theta. No Hessian/inner-graph terms.
            for p, total in zip(model.parameters(), accum):
                p.grad = total.div_(tasks)
            nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
            outer.step()
        else:
            epsilon = float(c["reptile_outer_lr"])  # fixed schedule also usable in fixed-step final refit
            with torch.no_grad():
                for p, total in zip(model.parameters(), accum):
                    p.add_(epsilon * (total.div_(tasks) - p))
        if step % int(c["eval_every"]) == 0 or step == steps:
            mae = _meta_validation(model, validation_tasks, c, device, seed) if validation_tasks is not None else None
            history.append({"step": step, "query_mse": float(np.mean(query_losses)), "validation_adapted_mae": mae})
            if mae is not None and mae < best_mae - 1e-9:
                best_mae, best_step, best_state = mae, step, _state(model)
            if checkpoint is not None:
                _torch_save(checkpoint, {"algorithm": algorithm, "state_dict": _state(model),
                    "optimizer": outer.state_dict() if outer is not None else None,
                    "step": step, "requested_steps": steps, "sampler_rng": sampler.rng.bit_generator.state,
                    "inner_rng": rng.bit_generator.state, "history": history, "episodes": episodes,
                    "best_state": best_state, "best_step": best_step, "best_mae": best_mae,
                    "elapsed": elapsed_before + time.perf_counter() - start})
    elapsed = elapsed_before + time.perf_counter() - start
    if select_best:
        model.load_state_dict(best_state)
    else:
        best_step = steps
    return FitResult(model, best_step, best_mae if validation_tasks is not None else None,
                     history, elapsed, episodes)


def _bundle(bundle: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    d = {key: canonical_frame(bundle[key], key) for key in ("donor_train", "donor_val", "donor_full")}
    d["targets"] = {str(city): {part: canonical_frame(value[part], f"{city}/{part}")
                              for part in ("adapt", "validation", "test")}
                    for city, value in bundle["targets"].items()}
    if not d["targets"]:
        raise ValueError("No target cities")
    donors = set(d["donor_full"].city)
    if donors & set(d["targets"]):
        raise ValueError("Selected target cities must be globally excluded from donors")
    if set(d["donor_train"].city) != donors or set(d["donor_val"].city) != donors:
        raise ValueError("Donor train/validation/full must contain the same cities")
    d["dev_cities"] = [str(v) for v in bundle.get("dev_cities", c["dev_cities"])]
    if not d["dev_cities"] or not set(d["dev_cities"]).issubset(donors):
        raise ValueError("dev_cities must be a nonempty subset of donor cities")
    if set(d["dev_cities"]) == donors:
        raise ValueError("Need donor cities outside development pseudo-tasks")
    for key, lo, hi in (("donor_train", "2020-01-01", "2022-12-31"),
                        ("donor_val", "2023-01-01", "2023-12-31"),
                        ("donor_full", "2020-01-01", "2023-12-31")):
        if not d[key].date.between(lo, hi).all():
            raise ValueError(f"{key}: label date outside declared window")
    if (d["donor_val"].issue_date < pd.Timestamp("2023-01-01")).any():
        raise ValueError("Donor validation issue dates must follow the 2022 fitting cutoff")
    # Final refit may also use the early-Jan-2023 labels excluded from the strict
    # donor-validation issue-date boundary; it must contain every source row unchanged.
    cols = list(REQUIRED)
    union = pd.concat([d["donor_train"], d["donor_val"]], ignore_index=True).sort_values(["city", "date", "horizon"])
    full = d["donor_full"].sort_values(["city", "date", "horizon"])
    union_hash = pd.util.hash_pandas_object(union[cols], index=False).to_numpy()
    full_hash = pd.util.hash_pandas_object(full[cols], index=False).to_numpy()
    if not np.isin(union_hash, full_hash).all():
        raise ValueError("donor_full must contain every donor_train and donor_val row unchanged")
    for city in donors:
        if len({str(d[key].loc[d[key].city == city, "station_id"].iloc[0])
                for key in ("donor_train", "donor_val", "donor_full")}) != 1:
            raise ValueError(f"{city}: donor station identity changes across periods")
    for city, parts in d["targets"].items():
        for part, frame in parts.items():
            if set(frame.city) != {city}:
                raise ValueError(f"{city}/{part}: city key and rows differ")
        if len({str(frame.station_id.iloc[0]) for frame in parts.values()}) != 1:
            raise ValueError(f"{city}: target station identity changes across periods")
        if not (parts["adapt"].date <= pd.Timestamp("2023-12-31")).all():
            raise ValueError("Target adaptation labels must be <= 2023-12-31")
        select_budget(parts["adapt"], max(c["budgets"]))
        for part, year in (("validation", 2024), ("test", 2025)):
            if not (parts[part].date.dt.year == year).all():
                raise ValueError(f"{city}/{part}: expected {year} labels")
            if (parts[part].issue_date < pd.Timestamp("2024-01-01")).any():
                raise ValueError(f"{city}/{part}: issue date must follow the actual 2023 fitting cutoff")
    return d


def _fingerprint(data: dict[str, Any], c: dict[str, Any]) -> str:
    digest = hashlib.sha256(ENGINE_VERSION.encode())
    # Resume and output preferences do not change scientific results.
    settings = {k: v for k, v in c.items() if k not in ("resume", "save_target_states")}
    digest.update(json.dumps(_jsonable(settings), sort_keys=True).encode())
    digest.update(json.dumps(data["dev_cities"]).encode())
    frames = [(key, data[key]) for key in ("donor_train", "donor_val", "donor_full")]
    frames += [(f"{city}/{part}", frame) for city, parts in sorted(data["targets"].items())
               for part, frame in sorted(parts.items())]
    for name, frame in frames:
        digest.update(name.encode())
        digest.update(pd.util.hash_pandas_object(frame[list(REQUIRED)], index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _fit_row(result: FitResult, *, stage: str, method: str, seed: int, train: pd.DataFrame,
              lr: float | None, requested_steps: int, city: str = "", budget: int | None = None) -> dict[str, Any]:
    return {"stage": stage, "method": method, "seed": seed, "city": city, "budget": budget,
        "architecture": result.model.architecture, "parameters": sum(p.numel() for p in result.model.parameters()),
        "train_examples": len(train), "train_label_dates": train.date.nunique(), "train_cities": train.city.nunique(),
        "train_first_label": train.date.min(), "train_last_label": train.date.max(), "lr": lr,
        "requested_steps": requested_steps, "selected_step": result.selected_step,
        "donor_validation_mae": result.validation_mae, "fit_seconds": result.elapsed,
        "selection_at_step_limit": (result.selected_step == requested_steps) if "selection" in stage else None,
        "selection_kept_persistence_initialization": (result.selected_step == 0) if "selection" in stage else None,
        "convergence_claimed": False,
        "selection_data": "donors only" if stage != "target_fit" else "preset frozen on donor pseudo-tasks",
        "target_validation_used": False}


def _save_fit(out: Path, name: str, result: FitResult, row: dict[str, Any]) -> None:
    _atomic_json(out / "fit_records" / f"{name}.json", row)
    _atomic_csv(out / "histories" / f"{name}.csv", pd.DataFrame(result.history))
    if result.episodes is not None:
        _atomic_csv(out / "meta_episodes" / f"{name}.csv.gz", pd.DataFrame(result.episodes))


def _prediction_rows(frame: pd.DataFrame, y_pred: np.ndarray, *, method: str, budget: int, seed: int,
                      period: str) -> pd.DataFrame:
    result = frame[["station_id", "city", "date", "issue_date", "horizon", "last"]].copy()
    result["budget"], result["seed"], result["method"] = budget, seed, method
    result["y_true"], result["y_pred"], result["period"] = frame.y.to_numpy(float), y_pred, period
    if len(y_pred) != len(result) or not np.isfinite(y_pred).all():
        raise ValueError("Prediction length/finiteness failure")
    for col in ("date", "issue_date"):
        result[col] = result[col].dt.strftime("%Y-%m-%d")
    return result[list(PREDICTION_COLUMNS)]


def _baseline(method: str, support: pd.DataFrame, queries: list[pd.DataFrame], seed: int,
              c: dict[str, Any], preset: dict[str, Any] | None = None
              ) -> tuple[list[np.ndarray], int | None, Any, float]:
    preset = preset or {}
    if method == "persistence":
        return [q["last"].to_numpy(float) for q in queries], 0, None, 0.0
    if method == "seasonal7":
        return [q[[f"x{i}" for i in range(7)]].to_numpy(float)[np.arange(len(q)), q.horizon.to_numpy(int) - 1]
                for q in queries], 0, None, 0.0
    if method == "mean7":
        return [q[[f"x{i}" for i in range(7)]].to_numpy(float).mean(axis=1) for q in queries], 0, None, 0.0
    if method == "ridge_local":
        model = Ridge(alpha=float(preset.get("alpha", c["ridge_alpha"])), solver="cholesky")
    elif method == "histgb_local":
        model = HistGradientBoostingRegressor(max_iter=int(c["histgb_iterations"]), learning_rate=0.05,
            max_leaf_nodes=7, min_samples_leaf=int(preset.get("min_samples_leaf", 10)),
            l2_regularization=float(preset.get("l2_regularization", 1.0)), early_stopping=False, random_state=seed)
    elif method == "randomforest_local":
        model = RandomForestRegressor(n_estimators=int(c["rf_trees"]), max_depth=preset.get("max_depth", 6),
                                      min_samples_leaf=int(preset.get("min_samples_leaf", 3)),
                                      max_features=1.0, n_jobs=int(c["num_threads"]), random_state=seed)
    else:
        raise ValueError(method)
    x = feature_array(support).reshape(len(support), -1)
    y = (support.y.to_numpy(float) - support["last"].to_numpy(float)) / 100.0
    start = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - start
    return [q["last"].to_numpy(float) + 100.0 * model.predict(feature_array(q).reshape(len(q), -1))
            for q in queries], (x.shape[1] + 1 if method == "ridge_local" else None), model, fit_seconds


def tune_baselines(tasks: list[tuple[str, int, pd.DataFrame, pd.DataFrame]], seed: int,
                    c: dict[str, Any], out: Path) -> dict[str, dict[str, Any]]:
    """Cheap, equally bounded donor-only tuning for the three local regressors."""
    settings = {}
    for method, grid in (("ridge_local", c["ridge_presets"]), ("histgb_local", c["histgb_presets"]),
                         ("randomforest_local", c["randomforest_presets"])):
        cache = out / "selection" / f"{method}.json"
        if cache.exists() and c["resume"]:
            settings[method] = json.loads(cache.read_text())["selected_preset"]
            continue
        scores = []
        for candidate, preset in enumerate(grid):
            for city, budget, support, query in tasks:
                predictions, parameters, _, seconds = _baseline(method, support, [query], seed, c, preset)
                errors = query[["horizon"]].copy()
                errors["error"] = np.abs(predictions[0] - query.y.to_numpy(float))
                scores.append({"selection": method, "candidate": candidate, "preset": json.dumps(preset, sort_keys=True),
                    "city": city, "budget": budget, "seed": seed, "architecture": method.removesuffix("_local"),
                    "parameters": parameters, "macro_horizon_mae": float(errors.groupby("horizon").error.mean().mean()),
                    "fit_seconds": seconds, "support_examples": len(support), "query_examples": len(query),
                    "support_first": support.date.min(), "support_last": support.date.max(),
                    "query_first": query.date.min(), "query_last": query.date.max(),
                    "purge_intervening_days": (query.date.min() - support.date.max()).days - 1})
        ranking = pd.DataFrame(scores).groupby("candidate", sort=True).macro_horizon_mae.mean()
        chosen = int(ranking.idxmin())
        settings[method] = copy.deepcopy(grid[chosen])
        _atomic_json(cache, {"selected_candidate": chosen, "selected_preset": settings[method],
            "mean_macro_mae": ranking.to_dict(), "scores": scores,
            "selection_source": "held-out donor-city 2023 pseudo-query; targets entirely excluded"})
    return settings


def _save_candidate_table(out: Path) -> None:
    scores = []
    for path in sorted((out / "selection").glob("*.json")):
        value = json.loads(path.read_text())
        for score in value["scores"]:
            row = {**score, "stage": "donor_pseudo_candidate", "method": score["selection"],
                   "selected_candidate": value["selected_candidate"],
                   "candidate_selected": int(score["candidate"]) == int(value["selected_candidate"]),
                   "target_validation_used": False}
            scores.append(row)
            name = f"candidate_{score['selection']}_c{score['candidate']}_{_slug(score['city'])}_b{score['budget']}_s{score['seed']}"
            _atomic_json(out / "fit_records" / f"{name}.json", row)
    _atomic_csv(out / "donor_candidate_scores.csv", pd.DataFrame(scores))


def _metadata(out: Path, data: dict[str, Any], c: dict[str, Any], device: torch.device) -> str:
    fingerprint = _fingerprint(data, c)
    path = out / "engine_metadata.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to a different bundle/config; choose a new output directory")
        if not c["resume"]:
            raise ValueError("Output directory already contains a run; use resume or a new output directory")
        return fingerprint
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scikit-learn", "torch")}
    _atomic_json(path, {"engine_version": ENGINE_VERSION, "fingerprint": fingerprint, "config": c,
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "device": str(device), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": True, "methods": METHODS, "dev_cities": data["dev_cities"],
        "donor_cities": sorted(data["donor_full"].city.unique()), "target_cities": sorted(data["targets"]),
        "scaling": {"aqi_inputs": "AQI / 500 (fixed)", "residual_target": "(y-last)/100 (fixed)",
                    "target_fitted_scalers": False, "output_clipping": False},
        "protocol": {"budget": "last B common label dates, three horizons, exactly 3B supervised examples",
            "past_context_outside_budget": "allowed; seven genuinely observed days ending at issue date",
            "source_selection": "non-development donor 2020-22 fitting; non-development donor 2023 uncapped MAE",
            "adaptation_selection": "held-out donor cities; 2020-22 support, purged 2023 queries; separate equal grids for scratch/fine-tuning",
            "source_refit": "all donor cities 2020-23, fixed number of steps selected on donors",
            "target_fitting": "labels <=2023-12-31; no target outcome-based stopping or tuning",
            "meta_selection": "donor 2023 MAE after fixed training-algorithm inner updates",
            "meta_first_order": "FOMAML averages query gradients at independently adapted parameter copies; no second derivatives",
            "meta_purge": "at least nine full intervening calendar days between support and query labels",
            "secondary_2024": "evaluation only, with issue dates >=2024-01-01",
            "test_2025": "frozen end-2023 weights with rolling observed inputs; model-aging evaluation"}})
    return fingerprint


def train_all(bundle: dict[str, Any], outdir: str | Path, config: dict[str, Any] | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all fixed methods, resumably; target queries never influence fitting.

    ``config`` may also carry runner metadata, retained verbatim in the frozen
    settings. Neural settings recognized here are listed in ``DEFAULTS``.
    Completed prediction blocks and source optimizer checkpoints are atomic.
    A failure is written to failure.json and re-raised; it never removes a method.
    """
    c = _config(config)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(c["num_threads"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    choice = c["device"]
    device = torch.device("cuda" if choice == "auto" and torch.cuda.is_available() else "cpu" if choice == "auto" else choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = _bundle(bundle, c)
    fingerprint = _metadata(out, data, c, device)
    started = time.perf_counter()
    try:
        return _run(data, out, c, device, fingerprint, started)
    except Exception as exc:
        _atomic_json(out / "failure.json", {"type": type(exc).__name__, "error": str(exc),
            "fingerprint": fingerprint, "elapsed_seconds_this_invocation": time.perf_counter() - started})
        raise


def _run(data: dict[str, Any], out: Path, c: dict[str, Any], device: torch.device,
          fingerprint: str, started: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = data["dev_cities"]
    preliminary_train = data["donor_train"][~data["donor_train"].city.isin(dev)].reset_index(drop=True)
    preliminary_val = data["donor_val"][~data["donor_val"].city.isin(dev)].reset_index(drop=True)
    seed0 = int(c["seeds"][0])
    pseudo_budgets = list(c["pseudo_budgets"] or c["budgets"])
    dev_tasks = _pseudo_tasks(data["donor_train"], data["donor_val"], dev, pseudo_budgets, c)
    meta_validation_tasks = _pseudo_tasks(data["donor_train"], data["donor_val"],
        sorted(preliminary_train.city.unique()), [int(c["meta_validation_budget"])], c)
    settings: dict[str, dict[str, Any]] = tune_baselines(dev_tasks, seed0, c, out)
    source_models: dict[tuple[str, int], ResidualForecaster] = {}
    stage = "source_models"
    for architecture in c["architectures"]:
        for seed in c["seeds"]:
            name = f"pooled_selection_{architecture}_s{seed}"
            print(f"[engine] donor selection {architecture} seed={seed}", flush=True)
            initial = _new_model(architecture, seed, c, device)
            selected = fit_neural(initial, preliminary_train, steps=int(c["pooled_steps"]),
                lr=float(c["pooled_lr"]), seed=seed, c=c, device=device, validation=preliminary_val,
                checkpoint=out / "checkpoints" / f"{name}.pt", select_best=True)
            _save_fit(out, name, selected, _fit_row(selected, stage="donor_checkpoint_selection", method=f"{architecture}_pooled",
                seed=seed, train=preliminary_train, lr=float(c["pooled_lr"]), requested_steps=int(c["pooled_steps"])))
            if seed == seed0:
                settings[f"{architecture}_local"], _ = tune_presets(architecture, None, dev_tasks,
                    c["scratch_presets"], seed=seed, c=c, device=device, label=f"{architecture}_local", out=out)
                settings[f"{architecture}_finetune"], _ = tune_presets(architecture, selected.model, dev_tasks,
                    c["adaptation_presets"], seed=seed, c=c, device=device, label=f"{architecture}_finetune", out=out)
            final_name = f"pooled_refit_{architecture}_s{seed}"
            final = fit_neural(_new_model(architecture, seed, c, device), data["donor_full"],
                steps=selected.selected_step, lr=float(c["pooled_lr"]), seed=seed, c=c, device=device,
                checkpoint=out / "checkpoints" / f"{final_name}.pt")
            row = _fit_row(final, stage="donor_fixed_refit", method=f"{architecture}_pooled", seed=seed,
                train=data["donor_full"], lr=float(c["pooled_lr"]), requested_steps=selected.selected_step)
            row["selected_source_validation_mae"] = selected.validation_mae
            _save_fit(out, final_name, final, row)
            _torch_save(out / "source_states" / f"{architecture}_s{seed}.pt", {"state_dict": _state(final.model),
                "architecture": architecture, "seed": seed, "parameters": row["parameters"],
                "selected_steps": selected.selected_step, "fingerprint": fingerprint})
            source_models[(architecture, seed)] = final.model.cpu()
            del initial, selected, final
    for algorithm in ("reptile", "fomaml"):
        for seed in c["seeds"]:
            name = f"{algorithm}_selection_s{seed}"
            print(f"[engine] donor meta selection {algorithm} seed={seed}", flush=True)
            selected = fit_meta(algorithm, preliminary_train, steps=int(c["meta_steps"]), seed=seed, c=c,
                device=device, validation_tasks=meta_validation_tasks,
                checkpoint=out / "checkpoints" / f"{name}.pt", select_best=True)
            _save_fit(out, name, selected, _fit_row(selected, stage="donor_meta_checkpoint_selection",
                method=f"{algorithm}_gru", seed=seed, train=preliminary_train,
                lr=float(c["meta_outer_lr"] if algorithm == "fomaml" else c["reptile_outer_lr"]), requested_steps=int(c["meta_steps"])))
            if seed == seed0:
                settings[f"{algorithm}_gru"], _ = tune_presets("gru", selected.model, dev_tasks,
                    c["adaptation_presets"], seed=seed, c=c, device=device, label=f"{algorithm}_gru", out=out)
            final_name = f"{algorithm}_refit_s{seed}"
            final = fit_meta(algorithm, data["donor_full"], steps=selected.selected_step, seed=seed, c=c,
                device=device, checkpoint=out / "checkpoints" / f"{final_name}.pt")
            row = _fit_row(final, stage="donor_meta_fixed_refit", method=f"{algorithm}_gru", seed=seed,
                train=data["donor_full"], lr=float(c["meta_outer_lr"] if algorithm == "fomaml" else c["reptile_outer_lr"]),
                requested_steps=selected.selected_step)
            row["selected_source_validation_mae"] = selected.validation_mae
            _save_fit(out, final_name, final, row)
            _torch_save(out / "source_states" / f"{algorithm}_s{seed}.pt", {"state_dict": _state(final.model),
                "architecture": "gru", "algorithm": algorithm, "seed": seed,
                "selected_steps": selected.selected_step, "fingerprint": fingerprint})
            source_models[(algorithm, seed)] = final.model.cpu()
            del selected, final
    _atomic_json(out / "frozen_adaptation_settings.json", settings)
    _save_candidate_table(out)
    methods = [*BASELINES, *(f"{a}_{m}" for a in c["architectures"] for m in ("local", "pooled_zero", "finetune")),
               "reptile_gru", "fomaml_gru"]
    _atomic_json(out / "progress.json", {"stage": "target_fits", "completed_blocks": 0, "methods": methods})
    completed, block_paths = 0, []
    for city, parts in sorted(data["targets"].items()):
        queries = [parts["validation"], parts["test"]]
        for budget in c["budgets"]:
            support = select_budget(parts["adapt"], budget)
            for seed in c["seeds"]:
                for method in methods:
                    name = f"{_slug(city)}_b{budget}_s{seed}_{method}"
                    path = out / "prediction_blocks" / f"{name}.csv.gz"
                    record = out / "fit_records" / f"target_{name}.json"
                    block_paths.append(path)
                    if path.exists() and record.exists() and c["resume"]:
                        completed += 1
                        continue
                    print(f"[engine] {city} B={budget} seed={seed} {method}", flush=True)
                    began = time.perf_counter()
                    model, fitted, params, preset = None, None, 0, None
                    fit_seconds = 0.0
                    if method in BASELINES:
                        predictions, params, fitted, fit_seconds = _baseline(method, support, queries, seed, c,
                            settings.get(method))
                    else:
                        if method in ("reptile_gru", "fomaml_gru"):
                            algorithm = method.removesuffix("_gru")
                            model = copy.deepcopy(source_models[(algorithm, seed)]).to(device)
                            preset = settings[method]
                        else:
                            architecture, mode = method.split("_", 1)
                            model = (_new_model(architecture, seed, c, device) if mode == "local" else
                                     copy.deepcopy(source_models[(architecture, seed)]).to(device))
                            preset = settings[method] if mode != "pooled_zero" else None
                        if preset is not None:
                            fitted = _adapt(model, support, preset, seed, c, device)
                            model = fitted.model
                            fit_seconds = fitted.elapsed
                        predictions = [predict_neural(model, q, device) for q in queries]
                        params = sum(p.numel() for p in model.parameters())
                    result = pd.concat([_prediction_rows(q, p, method=method, budget=budget, seed=seed, period=period)
                        for q, p, period in zip(queries, predictions, ("secondary_2024", "test_2025"))], ignore_index=True)
                    row = {"stage": "target_fit", "method": method, "seed": seed, "city": city, "budget": budget,
                        "architecture": model.architecture if model is not None else method.removesuffix("_local"),
                        "parameters": params, "train_examples": len(support) if method not in ("persistence", "seasonal7", "mean7")
                            and not method.endswith("pooled_zero") else 0,
                        "available_budget_examples": len(support), "train_label_dates": budget,
                        "train_first_label": support.date.min(), "train_last_label": support.date.max(),
                        "lr": preset["lr"] if preset else None, "requested_steps": preset["steps"] if preset else 0,
                        "selected_step": preset["steps"] if preset else 0, "target_validation_used": False,
                        "selection_data": "donor pseudo-tasks only; target schedule fixed", "fit_seconds": fit_seconds,
                        "block_seconds": time.perf_counter() - began,
                        "baseline_preset": json.dumps(settings[method], sort_keys=True) if method in settings and method in BASELINES else None,
                        "prediction_examples": len(result)}
                    if isinstance(fitted, FitResult):
                        _save_fit(out, f"target_{name}", fitted, row)
                    else:
                        _atomic_json(record, row)
                    if c["save_target_states"] and model is not None:
                        _torch_save(out / "target_states" / f"{name}.pt", {"state_dict": _state(model),
                            "fingerprint": fingerprint, "method": method, "city": city, "budget": budget,
                            "seed": seed, "preset": preset})
                    # A block appears complete only after all training artifacts have been saved.
                    _atomic_csv(path, result)
                    completed += 1
                    _atomic_json(out / "progress.json", {"stage": "target_fits", "completed_blocks": completed,
                        "total_blocks": len(data["targets"]) * len(c["budgets"]) * len(c["seeds"]) * len(methods),
                        "last_completed": name, "elapsed_seconds_this_invocation": time.perf_counter() - started})
                    del model, fitted, result
    predictions = pd.concat([pd.read_csv(path, dtype={"station_id": str}) for path in block_paths], ignore_index=True)
    fit_log = pd.DataFrame([json.loads(path.read_text()) for path in sorted((out / "fit_records").glob("*.json"))])
    diagnostics = fit_log[fit_log.stage.isin(["donor_checkpoint_selection", "donor_meta_checkpoint_selection"])].copy()
    _atomic_csv(out / "source_training_diagnostics.csv", diagnostics)
    expected = sum(len(p["validation"]) + len(p["test"]) for p in data["targets"].values()) * len(c["budgets"]) * len(c["seeds"]) * len(methods)
    if len(predictions) != expected or predictions.duplicated(["city", "period", "date", "horizon", "budget", "seed", "method"]).any():
        raise AssertionError("Prediction coverage or unique-key failure")
    _atomic_csv(out / "predictions.csv.gz", predictions)
    _atomic_csv(out / "fit_log.csv", fit_log)
    _atomic_json(out / "progress.json", {"stage": "complete", "completed_blocks": completed,
        "prediction_rows": len(predictions), "fit_rows": len(fit_log), "fingerprint": fingerprint,
        "elapsed_seconds_this_invocation": time.perf_counter() - started})
    failure = out / "failure.json"
    if failure.exists():
        failure.unlink()
    return predictions, fit_log


def _synthetic_frame(city: str, start: str, end: str, city_number: int) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=12), end)
    t = (dates - pd.Timestamp("2020-01-01")).days.to_numpy()
    values = 80 + city_number * 9 + 20 * np.sin(t * 2 * np.pi / 21) + 0.02 * (t % 365)
    series = pd.Series(values, index=dates)
    records = []
    # Label triples share a date, each horizon uses its own earlier issue window.
    for date in pd.date_range(start, end):
        for horizon in (1, 2, 3):
            issue = date - pd.Timedelta(days=horizon)
            lags = series.loc[pd.date_range(issue - pd.Timedelta(days=6), issue)].to_numpy(float)
            records.append({"station_id": f"synthetic_{city}", "city": city, "date": date, "issue_date": issue,
                "horizon": horizon, "y": float(series.loc[date]), "last": float(lags[-1]),
                **{f"x{i}": float(v) for i, v in enumerate(lags)}})
    return pd.DataFrame(records)


def smoke_test(out: str | Path) -> dict[str, Any]:
    """Artificial end-to-end run, gradient checks, leakage guards, and resume."""
    donor_train = pd.concat([_synthetic_frame(c, "2022-01-01", "2022-12-31", i)
                            for i, c in enumerate(("donor_a", "donor_b", "donor_dev"))], ignore_index=True)
    donor_val = pd.concat([_synthetic_frame(c, "2023-01-04", "2023-01-24", i)
                          for i, c in enumerate(("donor_a", "donor_b", "donor_dev"))], ignore_index=True)
    target = {"adapt": _synthetic_frame("target", "2023-10-01", "2023-12-31", 5),
              "validation": _synthetic_frame("target", "2024-01-04", "2024-01-12", 5),
              "test": _synthetic_frame("target", "2025-01-04", "2025-01-12", 5)}
    bundle = {"donor_train": donor_train, "donor_val": donor_val,
              "donor_full": pd.concat([donor_train, donor_val], ignore_index=True),
              "targets": {"target": target}, "dev_cities": ["donor_dev"]}
    config = {"smoke": True, "seeds": [17], "budgets": [5, 9], "hidden_size": 8,
        "pooled_steps": 3, "eval_every": 2, "meta_steps": 3, "meta_tasks": 2, "meta_inner_steps": 2,
        "meta_validation_budget": 5, "meta_query_days": 3, "pseudo_query_days": 6,
        "adaptation_presets": [{"lr": 0.001, "steps": 2}, {"lr": 0.003, "steps": 3}],
        "scratch_presets": [{"lr": 0.001, "steps": 2}, {"lr": 0.003, "steps": 3}],
        "batch_size": 16, "histgb_iterations": 3, "rf_trees": 4, "device": "cpu"}
    c = _config(config)
    support = canonical_frame(target["adapt"])
    assert set(select_budget(support, 5).date).issubset(set(select_budget(support, 9).date))
    for a in ARCHITECTURES:
        model = _new_model(a, 17, c, torch.device("cpu"))
        x = torch.as_tensor(feature_array(support.iloc[:9]))
        assert torch.equal(model(x), torch.zeros(9)), f"{a} residual initialization is not persistence"
        result = fit_neural(model, support.iloc[:9], steps=2, lr=0.003, seed=17, c=c, device=torch.device("cpu"))
        assert np.isfinite(predict_neural(result.model, support.iloc[:9], torch.device("cpu"))).all()
    # The meta update itself must change a zero-output initialization.
    preliminary = canonical_frame(donor_train[donor_train.city != "donor_dev"])
    for algorithm in ("reptile", "fomaml"):
        initial = _new_model("gru", 17, c, torch.device("cpu"))
        result = fit_meta(algorithm, preliminary, steps=2, seed=17, c=c, device=torch.device("cpu"))
        assert not torch.equal(initial.head.weight, result.model.head.weight), f"{algorithm} did not update"
        assert all(e["purge_intervening_days"] >= 9 for e in result.episodes)
    first, logs = train_all(bundle, out, config)
    assert set(first.method) == set(METHODS), "Missing method in smoke run"
    keysets = first.groupby(["method", "budget", "seed"]).apply(
        lambda g: frozenset(zip(g.period, g.date, g.horizon)), include_groups=False) if int(pd.__version__.split('.')[0]) >= 3 else None
    if keysets is not None:
        assert len(set(keysets)) == 1
    else:
        reference = None
        for _, group in first.groupby(["method", "budget", "seed"]):
            keys = set(zip(group.period, group.date, group.horizon))
            if reference is None:
                reference = keys
            assert keys == reference
    second, _ = train_all(bundle, out, config)
    pd.testing.assert_frame_equal(first, second)
    bad = copy.deepcopy(bundle)
    bad["targets"]["target"]["adapt"].loc[0, "date"] = pd.Timestamp("2024-01-01")
    rejected = False
    try:
        _bundle(bad, c)
    except ValueError:
        rejected = True
    assert rejected, "Leakage guard did not reject altered adaptation dates"
    summary = {"status": "passed", "artificial_data_only": True, "methods": len(first.method.unique()),
        "prediction_rows": len(first), "fit_records": len(logs), "resume_equal": True,
        "meta_algorithms_update": True, "purge_days_minimum": 9, "target_leakage_guard": True}
    _atomic_json(Path(out) / "smoke_test_result.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="/tmp/aqi-final-engine-smoke")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Use train_all from the frozen runner, or pass --smoke for synthetic verification")
    print(json.dumps(smoke_test(args.out), indent=2))
