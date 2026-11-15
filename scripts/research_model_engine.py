#!/usr/bin/env python3
"""Higher-capacity, donor-selected AQI transfer and second-order meta research.

Public API: ``predictions, fit_log = train_research(bundle, outdir, config)``.
The canonical input and prediction schema are identical to final_model_engine.
This module never uses target query labels for fitting, stopping or selection.
Source files are included in the run fingerprint; every fit is resumable.

``--smoke`` runs artificial-data tests only; ``--gradient-smoke`` explicitly
tests the Hessian contribution and body-weight updates of full MAML.
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
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

try:
    from scripts import final_model_engine as base
except ImportError:
    import final_model_engine as base


ENGINE_VERSION = "research-deep-maml-v1.0"
RESEARCH_SCHEMA_VERSION = 1
DEEP_ARCHITECTURES = ("gru", "lstm", "tcn", "transformer", "mlp")
BASELINES = base.BASELINES
METHODS = (*BASELINES,
    *(f"deep_{a}_{mode}" for a in DEEP_ARCHITECTURES for mode in ("local", "pooled_zero", "finetune")),
    *(f"matched_gru_{mode}" for mode in ("local", "pooled_zero", "finetune")),
    *(f"{a}_gru_{o}" for a in ("maml", "fomaml", "reptile") for o in ("sgd", "adam")),
    "diversity6_pooled_zero", "diversity6_finetune", "diversity6_maml_sgd")

DEFAULTS: dict[str, Any] = {
    **copy.deepcopy(base.DEFAULTS),
    "seeds": [17, 43, 71], "budgets": [30, 90, 180],
    "architectures": list(DEEP_ARCHITECTURES), "device": "auto", "num_threads": 2,
    "source_batch_size": 256, "batch_size": 256, "meta_batch_size": 32,
    "source_eval_every": 300, "source_min_steps": 600, "source_patience": 4,
    "meta_eval_every": 150, "meta_min_steps": 300, "meta_patience": 4,
    "source_grid": [
        {"hidden": 128, "lr": 0.001, "weight_decay": 0.0001, "steps": 3000},
        {"hidden": 128, "lr": 0.0003, "weight_decay": 0.0001, "steps": 6000},
        {"hidden": 192, "lr": 0.001, "weight_decay": 0.001, "steps": 3000},
        {"hidden": 192, "lr": 0.0003, "weight_decay": 0.001, "steps": 6000}],
    "meta_grid": [
        {"inner_lr": 0.01, "outer_lr": 0.001, "inner_steps": 3, "steps": 1000},
        {"inner_lr": 0.03, "outer_lr": 0.001, "inner_steps": 3, "steps": 1500},
        {"inner_lr": 0.01, "outer_lr": 0.0003, "inner_steps": 5, "steps": 1500},
        {"inner_lr": 0.03, "outer_lr": 0.0003, "inner_steps": 5, "steps": 1500}],
    "reptile_outer_scale": 100.0,
    "deep_scratch_grid": [
        {"optimizer": "adam", "lr": 0.001, "steps": 150, "weight_decay": 0.0001},
        {"optimizer": "adam", "lr": 0.0003, "steps": 150, "weight_decay": 0.0001},
        {"optimizer": "adam", "lr": 0.001, "steps": 400, "weight_decay": 0.001},
        {"optimizer": "adam", "lr": 0.0003, "steps": 400, "weight_decay": 0.001}],
    "deep_finetune_grid": [
        {"optimizer": "adam", "lr": 0.0003, "steps": 100, "weight_decay": 0.0001},
        {"optimizer": "adam", "lr": 0.001, "steps": 100, "weight_decay": 0.0001},
        {"optimizer": "adam", "lr": 0.0003, "steps": 300, "weight_decay": 0.001},
        {"optimizer": "adam", "lr": 0.001, "steps": 300, "weight_decay": 0.001}],
    # Same optimizer and equally sized search space for every matched main arm.
    "native_sgd_grid": [
        {"optimizer": "sgd", "lr": 0.01, "steps": 3, "weight_decay": 0.0},
        {"optimizer": "sgd", "lr": 0.03, "steps": 10, "weight_decay": 0.0},
        {"optimizer": "sgd", "lr": 0.01, "steps": 30, "weight_decay": 0.0},
        {"optimizer": "sgd", "lr": 0.03, "steps": 30, "weight_decay": 0.0}],
    "meta_adam_grid": [
        {"optimizer": "adam", "lr": 0.0003, "steps": 30, "weight_decay": 0.0},
        {"optimizer": "adam", "lr": 0.001, "steps": 30, "weight_decay": 0.0},
        {"optimizer": "adam", "lr": 0.0003, "steps": 100, "weight_decay": 0.0},
        {"optimizer": "adam", "lr": 0.001, "steps": 100, "weight_decay": 0.0}],
    "meta_tasks": 4, "meta_query_days": 14, "meta_validation_budget": 90,
    "manual_hidden": 64, "manual_layers": 2, "deep_layers": 2,
    "diversity_donors": 6, "grad_clip": 1.0,
    "hpo_development_weight": 0.5, "hpo_seed": 17,
    "source_hpo_adaptation_steps": 60, "pseudo_query_days": None,
    "compute_budget_seconds": 4 * 60 * 60 - 120,
    "checkpoint_every_seconds": 180, "resume": True, "save_target_states": False,
    "include_baselines": True, "include_meta": True, "include_diversity": True,
    "smoke": False,
}


class ComputeBudgetReached(RuntimeError):
    """A partial, resumable run; this is never reported as successful completion."""


def effective_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    c = copy.deepcopy(DEFAULTS)
    c.update(config or {})
    c["seeds"] = [int(v) for v in c["seeds"]]
    c["budgets"] = [int(v) for v in c["budgets"]]
    for key in ("seeds", "budgets"):
        if not c[key] or len(set(c[key])) != len(c[key]):
            raise ValueError(f"{key} must be nonempty without duplicates")
    if min(c["budgets"]) < 1 or int(c["purge_days"]) < 9:
        raise ValueError("Positive label budgets and >=9 full intervening purge days are required")
    if not set(c["architectures"]).issubset(DEEP_ARCHITECTURES):
        raise ValueError("Unknown deep architecture")
    for key in ("source_grid", "meta_grid", "deep_scratch_grid", "deep_finetune_grid", "native_sgd_grid", "meta_adam_grid"):
        if not c[key] or (not c["smoke"] and len(c[key]) != 4):
            raise ValueError(f"{key} must have four predeclared donor-only candidates")
        for p in c[key]:
            if int(p["steps"]) < 1:
                raise ValueError(f"{key}: steps must be positive")
    if any(int(p["steps"]) > 6000 for p in c["source_grid"]):
        raise ValueError("Each source candidate is capped at 6000 updates")
    if any(int(p["steps"]) > 1500 for p in c["meta_grid"]):
        raise ValueError("Each meta candidate is capped at 1500 outer updates")
    if int(c["manual_hidden"]) != 64 and not c["smoke"]:
        raise ValueError("Matched meta backbone is fixed at 64 hidden units")
    if int(c["manual_layers"]) != 2 and not c["smoke"]:
        raise ValueError("Matched meta backbone is fixed at two layers")
    for key in ("source_batch_size", "meta_batch_size", "meta_tasks", "source_eval_every", "meta_eval_every"):
        if int(c[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return c


def method_inventory(c: dict[str, Any]) -> list[str]:
    methods = list(BASELINES) if c["include_baselines"] else []
    methods += [f"deep_{a}_{m}" for a in c["architectures"] for m in ("local", "pooled_zero", "finetune")]
    methods += [f"matched_gru_{m}" for m in ("local", "pooled_zero", "finetune")]
    if c["include_meta"]:
        methods += [f"{a}_gru_{o}" for a in ("maml", "fomaml", "reptile") for o in ("sgd", "adam")]
    if c["include_diversity"]:
        methods += ["diversity6_pooled_zero", "diversity6_finetune"]
        if c["include_meta"]:
            methods += ["diversity6_maml_sgd"]
    return methods


def research_methods(config: dict[str, Any] | None = None) -> list[str]:
    """Exact predeclared inventory, available before loading any data."""
    return method_inventory(effective_config(config))


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CausalTCNBlock(nn.Module):
    def __init__(self, incoming: int, hidden: int, dilation: int):
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(incoming, hidden, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.shortcut = nn.Conv1d(incoming, hidden, 1) if incoming != hidden else nn.Identity()
        self.norm1, self.norm2 = nn.GroupNorm(1, hidden), nn.GroupNorm(1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm1(self.conv1(F.pad(x, (self.padding, 0)))))
        z = self.norm2(self.conv2(F.pad(z, (self.padding, 0))))
        return F.gelu(z + self.shortcut(x))


class DeepForecaster(nn.Module):
    """Residual forecast heads; inputs remain the same seven observed AQI days."""
    def __init__(self, architecture: str, hidden: int = 128):
        super().__init__()
        self.architecture, self.hidden = f"deep_{architecture}", hidden
        self.kind = architecture
        if architecture == "gru":
            self.body = nn.GRU(6, hidden, num_layers=2, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
        elif architecture == "lstm":
            self.body = nn.LSTM(6, hidden, num_layers=2, batch_first=True)
            self.attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.norm = nn.LayerNorm(hidden)
        elif architecture == "tcn":
            self.body = nn.Sequential(*(CausalTCNBlock(6 if j == 0 else hidden, hidden, 2 ** j) for j in range(4)))
        elif architecture == "transformer":
            if hidden % 8:
                raise ValueError("Deep Transformer hidden size must be divisible by eight")
            self.embed = nn.Linear(6, hidden)
            layer = nn.TransformerEncoderLayer(hidden, 8, hidden * 4, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
            positions = torch.arange(7).float()[:, None]
            div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
            pos = torch.zeros(7, hidden)
            pos[:, 0::2], pos[:, 1::2] = torch.sin(positions * div), torch.cos(positions * div)
            self.register_buffer("positions", pos[None])
        elif architecture == "mlp":
            self.embed = nn.Sequential(nn.Flatten(), nn.Linear(42, hidden), nn.GELU())
            self.body = nn.Sequential(*(ResidualMLPBlock(hidden) for _ in range(3)), nn.LayerNorm(hidden))
        else:
            raise ValueError(architecture)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gru":
            z = self.norm(self.body(x)[0][:, -1])
        elif self.kind == "lstm":
            seq = self.body(x)[0]
            z = self.norm((torch.softmax(self.attention(seq), dim=1) * seq).sum(dim=1))
        elif self.kind == "tcn":
            z = self.body(x.transpose(1, 2))[:, :, -1]
        elif self.kind == "transformer":
            z = self.body(self.embed(x) + self.positions)[:, -1]
        else:
            z = self.body(self.embed(x))
        return self.head(z).squeeze(-1)


class ManualGRULayer(nn.Module):
    """Explicit PyTorch-style reset/update/new gates, with double backward support."""
    def __init__(self, incoming: int, hidden: int):
        super().__init__()
        self.input_gates = nn.Linear(incoming, hidden * 3)
        self.hidden_gates = nn.Linear(hidden, hidden * 3)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.new_zeros((len(x), self.hidden))
        inputs = self.input_gates(x)
        values = []
        for t in range(x.shape[1]):
            ir, iz, inn = inputs[:, t].chunk(3, dim=-1)
            hr, hz, hn = self.hidden_gates(h).chunk(3, dim=-1)
            reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            h = (1 - update) * candidate + update * h
            values.append(h)
        return torch.stack(values, dim=1)


class ManualGRUForecaster(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.architecture, self.hidden = "matched_manual_gru", hidden
        self.layers = nn.ModuleList(ManualGRULayer(6 if j == 0 else hidden, hidden) for j in range(layers))
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, -1]).squeeze(-1)


def _new_model(spec: dict[str, Any], seed: int, device: torch.device) -> nn.Module:
    base._seed(seed)
    model = (ManualGRUForecaster(int(spec["hidden"]), int(spec.get("layers", 2))) if spec["architecture"] == "manual_gru"
             else DeepForecaster(spec["architecture"], int(spec["hidden"])))
    return model.to(device)


def _spec(architecture: str, candidate: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    if architecture == "manual_gru":
        return {"architecture": architecture, "hidden": int(c["manual_hidden"]), "layers": int(c["manual_layers"])}
    return {"architecture": architecture, "hidden": int(candidate["hidden"]),
        "layers": 4 if architecture == "tcn" else 3 if architecture in ("mlp", "transformer") else 2}


def _label_dates(frame: pd.DataFrame) -> list[str]:
    return [str(v)[:10] for v in np.sort(frame.date.unique())]


def _dates_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _prune(paths: Any) -> None:
    """Only discard recomputable optimizer files after their result is committed."""
    for path in paths:
        if path.exists():
            path.unlink()


def _train_fields(frame: pd.DataFrame) -> dict[str, Any]:
    dates = _label_dates(frame)
    return {"train_examples": len(frame), "train_label_dates": len(dates),
        "train_cities": frame.city.nunique(), "train_city_names": sorted(frame.city.unique()),
        "train_first_label": dates[0], "train_last_label": dates[-1],
        "train_label_dates_sha256": _dates_digest(dates), "target_validation_used": False}


def _deadline(c: dict[str, Any]) -> bool:
    return time.perf_counter() >= float(c.get("_deadline", float("inf")))


def _check_budget(c: dict[str, Any]) -> None:
    if _deadline(c):
        raise ComputeBudgetReached("Invocation compute budget reached; atomic checkpoints are available. Resume with the same code/config/data.")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _rng_state() -> dict[str, Any]:
    return {"torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(value: dict[str, Any]) -> None:
    torch.set_rng_state(value["torch"].cpu())
    if value.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in value["cuda"]])


def _fit_signature(model: nn.Module, train: pd.DataFrame, settings: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(base._jsonable(settings), sort_keys=True).encode())
    digest.update(str([(k, tuple(v.shape)) for k, v in model.state_dict().items()]).encode())
    digest.update(pd.util.hash_pandas_object(train[list(base.REQUIRED)], index=False).to_numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ResearchFit:
    model: nn.Module
    selected_step: int
    actual_steps: int
    validation_mae: float | None
    history: list[dict[str, Any]]
    elapsed: float
    stopped_early: bool = False
    episodes: list[dict[str, Any]] | None = None


def fit_supervised(model: nn.Module, train: pd.DataFrame, preset: dict[str, Any], *, seed: int,
                   c: dict[str, Any], device: torch.device, validation: pd.DataFrame | None = None,
                   checkpoint: Path | None = None, select_best: bool = False,
                   adaptation: bool = False) -> ResearchFit:
    """Balanced-city fixed-scale residual MSE; donor-only stopping uses raw AQI MAE."""
    if select_best and validation is None:
        raise ValueError("Selection requires donor validation")
    steps, lr = int(preset["steps"]), float(preset["lr"])
    algorithm, decay = preset.get("optimizer", "adam"), float(preset.get("weight_decay", 0.0))
    opt = (torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay) if algorithm == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay))
    data, sampler = base._arrays(train, device), base.CitySampler(train.city.to_numpy(str), seed + 991)
    signature = _fit_signature(model, train, {"preset": preset, "seed": seed, "adaptation": adaptation,
        "select_best": select_best, "eval_every": c["source_eval_every"], "min_steps": c["source_min_steps"]})
    first, elapsed_before, best_step, bad = 0, 0.0, 0, 0
    best_mae, best_state, history, stopped = float("inf"), base._state(model), [], False
    complete = False
    if checkpoint is not None and checkpoint.exists() and c["resume"]:
        value = base._torch_load(checkpoint, device)
        if value["signature"] != signature:
            raise ValueError(f"Fit signature mismatch: {checkpoint}")
        model.load_state_dict(value["state_dict"])
        opt.load_state_dict(value["optimizer"])
        sampler.rng.bit_generator.state = value["sampler_rng"]
        _restore_rng(value["rng"])
        first, elapsed_before = int(value["step"]), float(value["elapsed"])
        best_step, best_mae, best_state = int(value["best_step"]), float(value["best_mae"]), value["best_state"]
        history, bad = value["history"], int(value["bad_checks"])
        complete, stopped = bool(value["complete"]), bool(value["stopped_early"])
    _sync(device)
    began, saved_at, actual = time.perf_counter(), time.perf_counter(), first
    def save(done: bool) -> None:
        if checkpoint is not None:
            _sync(device)
            base._torch_save(checkpoint, {"signature": signature, "state_dict": base._state(model),
                "optimizer": opt.state_dict(), "sampler_rng": sampler.rng.bit_generator.state, "rng": _rng_state(),
                "step": actual, "best_step": best_step, "best_mae": best_mae, "best_state": best_state,
                "bad_checks": bad, "history": history, "complete": done, "stopped_early": stopped,
                "elapsed": elapsed_before + time.perf_counter() - began})
    if not complete:
        for step in range(first + 1, steps + 1):
            if _deadline(c):
                save(False)
                _check_budget(c)
            size = int(c["meta_batch_size"] if adaptation and algorithm == "sgd" else c["source_batch_size"])
            idx = torch.as_tensor(sampler.sample(size), device=device)
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(data.x[idx]), data.residual[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
            opt.step()
            actual = step
            check = step % int(c["source_eval_every"]) == 0 or step == steps
            if check:
                mae = base._macro_mae(model, validation, device) if validation is not None else None
                history.append({"step": step, "loss": float(loss.detach().cpu()), "validation_mae": mae})
                if mae is not None and step >= min(int(c["source_min_steps"]), steps):
                    if mae < best_mae - 1e-8:
                        best_mae, best_step, best_state, bad = mae, step, base._state(model), 0
                    else:
                        bad += 1
                    stopped = bool(select_best and bad >= int(c["source_patience"]))
                save(stopped or step == steps)
                saved_at = time.perf_counter()
                if stopped:
                    break
            elif time.perf_counter() - saved_at >= float(c["checkpoint_every_seconds"]):
                save(False)
                saved_at = time.perf_counter()
        save(True)
    _sync(device)
    elapsed = elapsed_before + time.perf_counter() - began
    if select_best:
        if best_step == 0:
            raise AssertionError("Donor selection must include a trained checkpoint")
        model.load_state_dict(best_state)
    return ResearchFit(model, best_step if select_best else actual, actual,
        best_mae if validation is not None else None, history, elapsed, stopped)


def _differentiable_inner(model: nn.Module, params: OrderedDict[str, torch.Tensor], x: torch.Tensor,
                          y: torch.Tensor, batches: list[np.ndarray], lr: float, *, second_order: bool,
                          grad_clip: float) -> OrderedDict[str, torch.Tensor]:
    """Explicit SGD; full MAML retains the complete support-gradient graph."""
    buffers = dict(model.named_buffers())
    for batch in batches:
        idx = torch.as_tensor(batch, device=x.device)
        loss = F.mse_loss(functional_call(model, (params, buffers), (x[idx],)), y[idx])
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=second_order)
        norm = torch.sqrt(sum(g.square().sum() for g in gradients) + 1e-12)
        scale = torch.clamp(torch.as_tensor(grad_clip, device=x.device) / norm, max=1.0)
        if not second_order:
            gradients = tuple(g.detach() for g in gradients)
            scale = scale.detach()
        params = OrderedDict((name, value - lr * scale * gradient)
            for (name, value), gradient in zip(params.items(), gradients))
    return params


def _meta_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batches: list[np.ndarray],
                   query: np.ndarray, preset: dict[str, Any], algorithm: str, grad_clip: float
                   ) -> tuple[list[torch.Tensor], float]:
    originals = OrderedDict(model.named_parameters())
    params = (OrderedDict((name, value.detach().clone().requires_grad_(True)) for name, value in originals.items())
              if algorithm == "reptile" else originals)
    adapted = _differentiable_inner(model, params, x, y, batches, float(preset["inner_lr"]),
        second_order=algorithm == "maml", grad_clip=grad_clip)
    qi = torch.as_tensor(query, device=x.device)
    query_loss = F.mse_loss(functional_call(model, (adapted, dict(model.named_buffers())), (x[qi],)), y[qi])
    if not torch.isfinite(query_loss):
        raise FloatingPointError("Nonfinite meta query loss")
    if algorithm == "reptile":
        gradients = [(original - adapted[name]).detach() for name, original in originals.items()]
    else:
        # FOMAML has identity paths through SGD but detached support gradients.
        # MAML differentiates through every SGD update, including the Hessian terms.
        gradients = [g.detach() for g in torch.autograd.grad(query_loss, tuple(originals.values()))]
    return gradients, float(query_loss.detach().cpu())


def _native_validation(model: nn.Module, tasks: list, preset: dict[str, Any], *, seed: int,
                       c: dict[str, Any], device: torch.device) -> float:
    scores = []
    for i, (_, _, support, query) in enumerate(tasks):
        _check_budget(c)
        adapted = copy.deepcopy(model).to(device)
        fitted = fit_supervised(adapted, support, {"optimizer": "sgd", "lr": preset["inner_lr"],
            "steps": preset["inner_steps"], "weight_decay": 0.0}, seed=seed + 37 * i, c=c, device=device,
            adaptation=True)
        scores.append(base._macro_mae(fitted.model, query, device))
    return float(np.mean(scores))


def fit_meta(algorithm: str, spec: dict[str, Any], train: pd.DataFrame, preset: dict[str, Any], *,
             seed: int, c: dict[str, Any], device: torch.device, validation_tasks: list | None = None,
             checkpoint: Path | None = None, select_best: bool = False,
             validation_group_boundary: int | None = None) -> ResearchFit:
    if algorithm not in ("maml", "fomaml", "reptile") or spec["architecture"] != "manual_gru":
        raise ValueError("All meta algorithms require the same explicit-gate GRU")
    if select_best and not validation_tasks:
        raise ValueError("Meta checkpoint selection requires donor pseudo-tasks")
    train = train.reset_index(drop=True)
    model = _new_model(spec, seed, device)
    data = base._arrays(train, device)
    sampler = base.EpisodeSampler(train, list(c["budgets"]), int(c["meta_query_days"]), int(c["purge_days"]), seed + 177)
    rng = np.random.default_rng(seed + 7919)
    steps, outer_lr = int(preset["steps"]), float(preset["outer_lr"])
    outer = torch.optim.Adam(model.parameters(), lr=outer_lr) if algorithm != "reptile" else None
    signature = _fit_signature(model, train, {"algorithm": algorithm, "preset": preset, "seed": seed,
        "spec": spec, "tasks": c["meta_tasks"], "meta_batch_size": c["meta_batch_size"],
        "select_best": select_best, "reptile_outer_scale": c["reptile_outer_scale"],
        "validation_group_boundary": validation_group_boundary})
    first, elapsed_before, best_step, bad, complete, stopped = 0, 0.0, 0, 0, False, False
    best_mae, best_state, history, episodes = float("inf"), base._state(model), [], []
    if checkpoint is not None and checkpoint.exists() and c["resume"]:
        value = base._torch_load(checkpoint, device)
        if value["signature"] != signature:
            raise ValueError(f"Meta fit signature mismatch: {checkpoint}")
        model.load_state_dict(value["state_dict"])
        if outer is not None:
            outer.load_state_dict(value["optimizer"])
        sampler.rng.bit_generator.state, rng.bit_generator.state = value["sampler_rng"], value["inner_rng"]
        _restore_rng(value["rng"])
        first, elapsed_before = int(value["step"]), float(value["elapsed"])
        best_step, best_mae, best_state = int(value["best_step"]), float(value["best_mae"]), value["best_state"]
        history, episodes, bad = value["history"], value["episodes"], int(value["bad_checks"])
        complete, stopped = bool(value["complete"]), bool(value["stopped_early"])
    _sync(device)
    began, saved_at, actual = time.perf_counter(), time.perf_counter(), first
    def save(done: bool) -> None:
        if checkpoint is not None:
            _sync(device)
            base._torch_save(checkpoint, {"signature": signature, "state_dict": base._state(model),
                "optimizer": outer.state_dict() if outer is not None else None,
                "sampler_rng": sampler.rng.bit_generator.state, "inner_rng": rng.bit_generator.state,
                "rng": _rng_state(), "step": actual, "best_step": best_step, "best_mae": best_mae,
                "best_state": best_state, "history": history, "episodes": episodes, "bad_checks": bad,
                "complete": done, "stopped_early": stopped, "elapsed": elapsed_before + time.perf_counter() - began})
    if not complete:
        for step in range(first + 1, steps + 1):
            if _deadline(c):
                save(False)
                _check_budget(c)
            totals = [torch.zeros_like(p) for p in model.parameters()]
            losses = []
            model.train()
            for task in range(int(c["meta_tasks"])):
                si, qi, audit = sampler.sample()
                batches = [rng.choice(si, size=min(int(c["meta_batch_size"]), len(si)), replace=True)
                           for _ in range(int(preset["inner_steps"]))]
                query = rng.choice(qi, size=min(int(c["meta_batch_size"]), len(qi)), replace=False)
                gradients, loss = _meta_gradient(model, data.x, data.residual, batches, query, preset,
                    algorithm, float(c["grad_clip"]))
                for total, gradient in zip(totals, gradients):
                    total.add_(gradient)
                losses.append(loss)
                sd = sorted(set(str(v)[:10] for v in data.dates[si]))
                qd = sorted(set(str(v)[:10] for v in data.dates[qi]))
                episodes.append({**audit, "step": step, "task": task, "seed": seed, "algorithm": algorithm,
                    "support_label_dates_sha256": _dates_digest(sd), "query_label_dates_sha256": _dates_digest(qd),
                    "inner_steps": int(preset["inner_steps"]), "inner_lr": float(preset["inner_lr"]),
                    "support_batch_examples": len(batches[0]), "query_batch_examples": len(query),
                    "full_second_order": algorithm == "maml"})
            if algorithm == "reptile":
                epsilon = outer_lr * float(c["reptile_outer_scale"])
                with torch.no_grad():
                    for parameter, total in zip(model.parameters(), totals):
                        parameter.add_(-epsilon * total / int(c["meta_tasks"]))
            else:
                outer.zero_grad(set_to_none=True)
                for parameter, total in zip(model.parameters(), totals):
                    parameter.grad = total / int(c["meta_tasks"])
                nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
                outer.step()
            actual = step
            if step % int(c["meta_eval_every"]) == 0 or step == steps:
                if validation_tasks and validation_group_boundary is not None:
                    boundary = int(validation_group_boundary)
                    chronological_mae = _native_validation(model, validation_tasks[:boundary], preset,
                        seed=seed, c=c, device=device)
                    development_mae = _native_validation(model, validation_tasks[boundary:], preset,
                        seed=seed, c=c, device=device)
                    w = float(c["hpo_development_weight"])
                    mae = (1 - w) * chronological_mae + w * development_mae
                else:
                    mae = _native_validation(model, validation_tasks, preset, seed=seed, c=c, device=device) if validation_tasks else None
                history.append({"step": step, "query_mse": float(np.mean(losses)), "validation_adapted_mae": mae})
                if mae is not None and step >= min(int(c["meta_min_steps"]), steps):
                    if mae < best_mae - 1e-8:
                        best_mae, best_step, best_state, bad = mae, step, base._state(model), 0
                    else:
                        bad += 1
                    stopped = bool(select_best and bad >= int(c["meta_patience"]))
                save(stopped or step == steps)
                saved_at = time.perf_counter()
                if stopped:
                    break
            elif time.perf_counter() - saved_at >= float(c["checkpoint_every_seconds"]):
                save(False)
                saved_at = time.perf_counter()
        save(True)
    _sync(device)
    elapsed = elapsed_before + time.perf_counter() - began
    if select_best:
        if best_step == 0:
            raise AssertionError("Meta selection must retain a trained checkpoint")
        model.load_state_dict(best_state)
    return ResearchFit(model, best_step if select_best else actual, actual,
        best_mae if validation_tasks else None, history, elapsed, stopped, episodes)


def _cost(model: nn.Module) -> dict[str, Any]:
    params = sum(v.numel() for v in model.parameters())
    # Explicitly a proxy, not a profiler measurement. Sequence models reuse weights 7 times.
    reuse = 1 if model.architecture == "deep_mlp" else 7
    return {"parameters": params, "forward_flops_proxy": 2 * params * reuse,
        "flops_estimator": "2*parameters*weight_reuse (7 sequence positions, 1 for MLP); excludes nonlinearities, attention matrices and optimizer",
        "estimated_float32_parameter_bytes": params * 4}


def _record_fit(out: Path, name: str, result: ResearchFit, train: pd.DataFrame, *, stage: str,
                method: str, seed: int, preset: dict[str, Any], spec: dict[str, Any],
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"stage": stage, "method": method, "seed": seed, "architecture": result.model.architecture,
        "model_spec": spec, "optimizer": preset.get("optimizer", "outer_adam"),
        "lr": preset.get("lr", preset.get("outer_lr")), "requested_steps": int(preset["steps"]),
        "selected_step": result.selected_step, "actual_steps": result.actual_steps,
        "fit_seconds": result.elapsed, "device_fit_seconds": result.elapsed,
        "stopped_early": result.stopped_early, "selection_at_step_limit": result.selected_step == int(preset["steps"]),
        "donor_validation_mae": result.validation_mae, "convergence_claimed": False,
        "selection_data": "donor cities only", **_train_fields(train), **_cost(result.model), **(extra or {})}
    if result.episodes is not None:
        path = Path("meta_episodes") / f"{name}.csv.gz"
        base._atomic_csv(out / path, pd.DataFrame(result.episodes))
        row["episodes_file"] = str(path)
        row["episode_rows"] = len(result.episodes)
    base._atomic_json(out / "fit_records" / f"{name}.json", row)
    base._atomic_json(out / "training_histories" / f"{name}.json", result.history)
    base._atomic_json(out / "progress.json", {"stage": stage, "last_completed_fit": name,
        "selected_step": result.selected_step, "actual_steps": result.actual_steps,
        "fit_seconds": result.elapsed, "train_cities": row["train_cities"]})
    return row


def _adaptation_tune(out: Path, label: str, spec: dict[str, Any], initial: nn.Module | None,
                     tasks: list, presets: list[dict[str, Any]], *, seed: int,
                     c: dict[str, Any], device: torch.device) -> dict[str, Any]:
    path = out / "selection" / f"adapt_{label}.json"
    if path.exists() and c["resume"]:
        _prune((out / "checkpoints").glob(f"adapt_{label}_c*_t*.pt"))
        return json.loads(path.read_text())["selected_preset"]
    scores = []
    for candidate, preset in enumerate(presets):
        for task, (city, budget, support, query) in enumerate(tasks):
            _check_budget(c)
            name = f"adapt_{label}_c{candidate}_t{task}"
            model = _new_model(spec, seed, device) if initial is None else copy.deepcopy(initial).to(device)
            fit = fit_supervised(model, support, preset, seed=seed + task * 37, c=c, device=device,
                checkpoint=out / "checkpoints" / f"{name}.pt", adaptation=True)
            score = base._macro_mae(fit.model, query, device)
            row = {"selection": label, "candidate": candidate, "city": city, "budget": budget,
                "macro_horizon_mae": score, "query_first": query.date.min(), "query_last": query.date.max(),
                "purge_intervening_days": (query.date.min() - support.date.max()).days - 1}
            _record_fit(out, name, fit, support, stage="donor_adaptation_candidate", method=label, seed=seed,
                preset=preset, spec=spec, extra=row)
            scores.append(row)
    rank = pd.DataFrame(scores).groupby("candidate", sort=True).macro_horizon_mae.mean()
    chosen = int(rank.idxmin())
    result = {"selected_candidate": chosen, "selected_preset": presets[chosen], "scores": scores,
        "selection_source": "heldout development donor cities, 2020-22 support and purged 2023 query; no target outcomes"}
    base._atomic_json(path, result)
    _prune((out / "checkpoints").glob(f"adapt_{label}_c*_t*.pt"))
    return copy.deepcopy(presets[chosen])


def _source_select(out: Path, architecture: str, train: pd.DataFrame, validation: pd.DataFrame,
                   dev_tasks: list, *, c: dict[str, Any], device: torch.device
                   ) -> tuple[dict[str, Any], dict[str, Any], int, nn.Module]:
    path = out / "selection" / f"source_{architecture}.json"
    seed, trials = int(c["hpo_seed"]), []
    selected = json.loads(path.read_text()) if path.exists() and c["resume"] else None
    candidates = [int(selected["selected_candidate"])] if selected is not None else range(len(c["source_grid"]))
    fits = {}
    for candidate in candidates:
        p = copy.deepcopy(c["source_grid"][candidate])
        spec = _spec(architecture, p, c)
        preset = {"optimizer": "adam", "lr": p["lr"], "steps": p["steps"], "weight_decay": p["weight_decay"]}
        name = f"hpo_source_{architecture}_c{candidate}"
        print(f"[research] source HPO {architecture} candidate={candidate} cap={p['steps']}", flush=True)
        fit = fit_supervised(_new_model(spec, seed, device), train, preset, seed=seed, c=c, device=device,
            validation=validation, select_best=True, checkpoint=out / "checkpoints" / f"{name}.pt")
        fits[candidate] = fit
        if selected is not None:
            _prune((out / "checkpoints").glob(f"hpo_source_{architecture}_c*_development_t*.pt"))
            return selected["model_spec"], selected["selected_preset"], int(selected["selected_steps"]), fit.model
        calibration = ({"optimizer": "sgd", "lr": 0.01, "steps": 3, "weight_decay": 0.0} if architecture == "manual_gru"
                       else {"optimizer": "adam", "lr": 0.0003, "steps": int(c["source_hpo_adaptation_steps"]), "weight_decay": 0.0001})
        dev_scores = []
        for i, (_, _, support, query) in enumerate(dev_tasks):
            calibration_name = f"{name}_development_t{i}"
            tuned = fit_supervised(copy.deepcopy(fit.model), support, calibration, seed=seed + 37 * i,
                c=c, device=device, adaptation=True,
                checkpoint=out / "checkpoints" / f"{calibration_name}.pt")
            development_score = base._macro_mae(tuned.model, query, device)
            dev_scores.append(development_score)
            _record_fit(out, calibration_name, tuned, support, stage="donor_hpo_development", method=architecture,
                seed=seed, preset=calibration, spec=spec, extra={"source_candidate": candidate,
                    "development_task": i, "macro_horizon_mae": development_score,
                    "query_first": query.date.min(), "query_last": query.date.max(),
                    "purge_intervening_days": (query.date.min() - support.date.max()).days - 1})
        weight = float(c["hpo_development_weight"])
        score = (1 - weight) * float(fit.validation_mae) + weight * float(np.mean(dev_scores))
        row = {"candidate": candidate, "score": score, "chronological_donor_mae": fit.validation_mae,
            "heldout_development_adapted_mae": float(np.mean(dev_scores)), "development_weight": weight,
            "model_spec": spec, "preset": preset, "selected_steps": fit.selected_step}
        trials.append(row)
        _record_fit(out, name, fit, train, stage="donor_hpo_source", method=architecture, seed=seed,
            preset=preset, spec=spec, extra={"candidate": candidate, "hpo_score": score,
                "heldout_development_adapted_mae": float(np.mean(dev_scores)), "validation_city_names": sorted(validation.city.unique())})
        base._atomic_json(out / "selection" / f"source_{architecture}_trials.json", {"trials": trials})
    chosen = min(trials, key=lambda v: (v["score"], v["candidate"]))
    selected = {"selected_candidate": chosen["candidate"], "model_spec": chosen["model_spec"],
        "selected_preset": chosen["preset"], "selected_steps": chosen["selected_steps"], "trials": trials,
        "selection_source": "nondevelopment donor 2020-22 fit; chronological donor 2023 MAE and heldout development-city adaptation MAE, equal weight"}
    base._atomic_json(path, selected)
    _prune((out / "checkpoints").glob(f"hpo_source_{architecture}_c*_development_t*.pt"))
    _prune(out / "checkpoints" / f"hpo_source_{architecture}_c{i}.pt"
           for i in range(len(c["source_grid"])) if i != int(chosen["candidate"]))
    return chosen["model_spec"], chosen["preset"], int(chosen["selected_steps"]), fits[chosen["candidate"]].model


def _meta_select(out: Path, algorithm: str, spec: dict[str, Any], train: pd.DataFrame,
                 chronological_tasks: list, dev_tasks: list, *, c: dict[str, Any], device: torch.device
                 ) -> tuple[dict[str, Any], int, nn.Module]:
    path = out / "selection" / f"meta_{algorithm}.json"
    seed, trials, fits = int(c["hpo_seed"]), [], {}
    selected = json.loads(path.read_text()) if path.exists() and c["resume"] else None
    candidates = [int(selected["selected_candidate"])] if selected is not None else range(len(c["meta_grid"]))
    # Duplicate groups only for checkpoint selection is avoided: each chronological
    # task and each development task enters once; final HPO score balances groups.
    validation_tasks = chronological_tasks + dev_tasks
    for candidate in candidates:
        preset = copy.deepcopy(c["meta_grid"][candidate])
        name = f"hpo_meta_{algorithm}_c{candidate}"
        print(f"[research] meta HPO {algorithm} candidate={candidate} cap={preset['steps']}", flush=True)
        fit = fit_meta(algorithm, spec, train, preset, seed=seed, c=c, device=device,
            validation_tasks=validation_tasks, select_best=True, checkpoint=out / "checkpoints" / f"{name}.pt",
            validation_group_boundary=len(chronological_tasks))
        fits[candidate] = fit
        if selected is not None:
            return selected["selected_preset"], int(selected["selected_steps"]), fit.model
        chronological = _native_validation(fit.model, chronological_tasks, preset, seed=seed, c=c, device=device)
        development = _native_validation(fit.model, dev_tasks, preset, seed=seed, c=c, device=device)
        w = float(c["hpo_development_weight"])
        score = (1 - w) * chronological + w * development
        row = {"candidate": candidate, "score": score, "chronological_donor_adapted_mae": chronological,
            "heldout_development_adapted_mae": development, "preset": preset, "selected_steps": fit.selected_step}
        trials.append(row)
        _record_fit(out, name, fit, train, stage="donor_hpo_meta", method=algorithm, seed=seed,
            preset=preset, spec=spec, extra={"candidate": candidate, "algorithm": algorithm,
                "full_second_order": algorithm == "maml", "hpo_score": score,
                "chronological_donor_adapted_mae": chronological, "heldout_development_adapted_mae": development})
        base._atomic_json(out / "selection" / f"meta_{algorithm}_trials.json", {"trials": trials})
    chosen = min(trials, key=lambda v: (v["score"], v["candidate"]))
    selected = {"selected_candidate": chosen["candidate"], "selected_preset": chosen["preset"],
        "selected_steps": chosen["selected_steps"], "trials": trials, "model_spec": spec,
        "selection_source": "only donor 2023 queries after native SGD; equal chronological/development group weights"}
    base._atomic_json(path, selected)
    _prune(out / "checkpoints" / f"hpo_meta_{algorithm}_c{i}.pt"
           for i in range(len(c["meta_grid"])) if i != int(chosen["candidate"]))
    return chosen["preset"], int(chosen["selected_steps"]), fits[chosen["candidate"]].model


def _source_hash(module: Any) -> str:
    path = getattr(module, "__file__", None)
    if path and Path(path).is_file():
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    text = getattr(module, "EMBEDDED_SOURCE", None)
    if text is None:
        raise RuntimeError("Embedded base module must expose __file__ or EMBEDDED_SOURCE for code fingerprinting")
    return hashlib.sha256(text.encode()).hexdigest()


def _metadata(out: Path, data: dict[str, Any], c: dict[str, Any], device: torch.device) -> str:
    public = {k: v for k, v in c.items() if not k.startswith("_")}
    code_hashes = {"research_model_engine.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "final_model_engine.py": _source_hash(base)}
    runtime_identity = {"torch_version": torch.__version__, "device_type": device.type,
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    digest = hashlib.sha256(ENGINE_VERSION.encode())
    digest.update(base._fingerprint(data, public).encode())
    digest.update(json.dumps(code_hashes, sort_keys=True).encode())
    # A resumed research run cannot silently combine CPU, MPS and CUDA blocks.
    digest.update(json.dumps(runtime_identity, sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    path = out / "engine_metadata.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Research output belongs to different code/config/data; choose a new output directory")
        if not c["resume"]:
            raise ValueError("Output exists; resume or choose another directory")
        return fingerprint
    meta_spec = _spec("manual_gru", {}, c)
    base._atomic_json(path, {"research_schema_version": RESEARCH_SCHEMA_VERSION, "engine_version": ENGINE_VERSION,
        "fingerprint": fingerprint, "code_sha256": code_hashes, "config": public,
        "runtime_identity": runtime_identity,
        "methods": method_inventory(c), "model_specs": {"meta_and_matched": meta_spec,
            "deep_candidates": {a: [_spec(a, p, c) for p in c["source_grid"]] for a in c["architectures"]}},
        "python": sys.version, "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "torch")},
        "device": str(device), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "donor_cities": sorted(data["donor_full"].city.unique()), "dev_cities": data["dev_cities"],
        "target_cities": sorted(data["targets"]),
        "scaling": {"aqi_inputs": "AQI/500 fixed", "residual_target": "(y-last)/100 fixed",
            "target_fitted_scalers": False, "output_clipping": False},
        "protocol": {"budget": "last B common label dates <=2023, exactly 3B supervised examples; 7 observed past days allowed",
            "source_hpo": "4 candidates, nondevelopment donor 2020-22 training, chronological donor 2023 plus heldout development adaptation",
            "source_refit": "fixed donor-selected updates on donor 2020-23; all target cities excluded",
            "meta": "manual explicit-gate GRU64x2; full MAML differentiates every support SGD step; FOMAML detaches support gradients; Reptile parameter interpolation",
            "meta_primary_adaptation": "native SGD; equal four-preset donor-only search for matched local/FT and every meta algorithm",
            "meta_adam_ablation": "same meta initialization; donor-selected Adam adaptation is secondary optimizer-mismatch analysis",
            "diversity": "donor-pool-size (city diversity and unique-data volume together), deterministic six nondevelopment donors; matched backbone/gradient update counts/all-donor-selected hyperparameters; not isolated causal task diversity",
            "purge": "at least nine full intervening days between support and query labels",
            "evaluation": "secondary2024/main2025; frozen2023 weights, rolling observed inputs; target queries never influence settings",
            "early_stop": "donor-only MAE, min600 source/min300 meta, patience4; meta checkpoint/ranking both use equal chronological/development group weights; fixed-step final refit; no convergence claim",
            "hpo_randomness": "one predeclared HPO seed17; selected hyperparameters and duration reused for final seeds17/43/71",
            "compute": "hard invocation wall-time budget with resumable atomic checkpoints; CUDA preferred, CPU supported"}})
    return fingerprint


def _choose_device(c: dict[str, Any]) -> torch.device:
    choice = c["device"]
    # MPS is opt-in: synthetic double-backward is validated below before accepting it.
    device = torch.device("cuda" if choice == "auto" and torch.cuda.is_available() else "cpu" if choice == "auto" else choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def train_research(bundle: dict[str, Any], outdir: str | Path, config: dict[str, Any] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = effective_config(config)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(c["num_threads"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    device = _choose_device(c)
    data = base._bundle(bundle, c)
    fingerprint = _metadata(out, data, c, device)
    started = time.perf_counter()
    c["_deadline"] = started + float(c["compute_budget_seconds"])
    base._atomic_json(out / "effective_config.json", {k: v for k, v in c.items() if not k.startswith("_")})
    base._atomic_json(out / "frozen_methods.json", {"methods": method_inventory(c), "fingerprint": fingerprint})
    try:
        # A genuine second derivative is checked on the selected backend before training.
        gradient = gradient_smoke(device=device)
        base._atomic_json(out / "second_order_gradient_check.json", gradient)
        return _run(data, out, c, device, fingerprint, started)
    except Exception as exc:
        base._atomic_json(out / "failure.json", {"type": type(exc).__name__, "error": str(exc),
            "fingerprint": fingerprint, "resumable_budget_pause": isinstance(exc, ComputeBudgetReached),
            "elapsed_seconds_this_invocation": time.perf_counter() - started})
        raise


# Keep the original runner's entry point available for embedding compatibility.
train_all = train_research


def _run(data: dict[str, Any], out: Path, c: dict[str, Any], device: torch.device,
          fingerprint: str, started: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = data["dev_cities"]
    train = data["donor_train"][~data["donor_train"].city.isin(dev)].reset_index(drop=True)
    validation = data["donor_val"][~data["donor_val"].city.isin(dev)].reset_index(drop=True)
    pseudo_budgets = list(c["pseudo_budgets"] or c["budgets"])
    dev_tasks = base._pseudo_tasks(data["donor_train"], data["donor_val"], dev, pseudo_budgets, c)
    chronological_tasks = base._pseudo_tasks(data["donor_train"], data["donor_val"],
        sorted(train.city.unique()), [int(c["meta_validation_budget"])], c)
    seed0 = int(c["hpo_seed"])
    settings = base.tune_baselines(dev_tasks, seed0, c, out) if c["include_baselines"] else {}
    specs, sources, selected_sources = {}, {}, {}
    for architecture in [*c["architectures"], "manual_gru"]:
        _check_budget(c)
        spec, source_preset, steps, preliminary = _source_select(out, architecture, train, validation, dev_tasks, c=c, device=device)
        specs[architecture] = spec
        selected_sources[architecture] = {"preset": source_preset, "steps": steps}
        prefix = "matched_gru" if architecture == "manual_gru" else f"deep_{architecture}"
        local_grid = c["native_sgd_grid"] if architecture == "manual_gru" else c["deep_scratch_grid"]
        ft_grid = c["native_sgd_grid"] if architecture == "manual_gru" else c["deep_finetune_grid"]
        settings[f"{prefix}_local"] = _adaptation_tune(out, f"{prefix}_local", spec, None, dev_tasks,
            local_grid, seed=seed0, c=c, device=device)
        settings[f"{prefix}_finetune"] = _adaptation_tune(out, f"{prefix}_finetune", spec, preliminary, dev_tasks,
            ft_grid, seed=seed0, c=c, device=device)
        del preliminary
        for seed in c["seeds"]:
            name = f"refit_source_{architecture}_s{seed}"
            preset = {**source_preset, "steps": steps}
            fit = fit_supervised(_new_model(spec, seed, device), data["donor_full"], preset, seed=seed, c=c,
                device=device, checkpoint=out / "checkpoints" / f"{name}.pt")
            _record_fit(out, name, fit, data["donor_full"], stage="donor_fixed_refit", method=prefix,
                seed=seed, preset=preset, spec=spec)
            sources[(architecture, seed)] = fit.model.cpu()
            base._torch_save(out / "source_states" / f"{architecture}_s{seed}.pt", {"state_dict": base._state(fit.model),
                "model_spec": spec, "preset": preset, "fingerprint": fingerprint})
    meta_selected = {}
    if c["include_meta"]:
        spec = specs["manual_gru"]
        for algorithm in ("maml", "fomaml", "reptile"):
            preset, steps, preliminary = _meta_select(out, algorithm, spec, train, chronological_tasks, dev_tasks, c=c, device=device)
            meta_selected[algorithm] = {"preset": preset, "steps": steps}
            for optimizer in ("sgd", "adam"):
                label = f"{algorithm}_gru_{optimizer}"
                settings[label] = _adaptation_tune(out, label, spec, preliminary, dev_tasks,
                    c["native_sgd_grid"] if optimizer == "sgd" else c["meta_adam_grid"], seed=seed0, c=c, device=device)
            del preliminary
            for seed in c["seeds"]:
                name = f"refit_meta_{algorithm}_s{seed}"
                chosen = {**preset, "steps": steps}
                fit = fit_meta(algorithm, spec, data["donor_full"], chosen, seed=seed, c=c, device=device,
                    checkpoint=out / "checkpoints" / f"{name}.pt")
                _record_fit(out, name, fit, data["donor_full"], stage="donor_meta_fixed_refit", method=algorithm,
                    seed=seed, preset=chosen, spec=spec, extra={"algorithm": algorithm, "full_second_order": algorithm == "maml"})
                sources[(algorithm, seed)] = fit.model.cpu()
                base._torch_save(out / "source_states" / f"{algorithm}_s{seed}.pt", {"state_dict": base._state(fit.model),
                    "model_spec": spec, "preset": chosen, "fingerprint": fingerprint})
    if c["include_diversity"]:
        # Stable, outcome-independent city identities. Development cities never enter
        # this restricted source; all other settings are copied from the all-donor arm.
        ordered = sorted(train.city.unique(), key=lambda v: hashlib.sha256(f"diversity6:{v}".encode()).hexdigest())
        count = min(int(c["diversity_donors"]), len(ordered))
        if count != int(c["diversity_donors"]) and not c["smoke"]:
            raise ValueError("Insufficient preliminary donor cities for diversity6 ablation")
        cities = sorted(ordered[:count])
        restricted = data["donor_full"][data["donor_full"].city.isin(cities)].reset_index(drop=True)
        base._atomic_json(out / "diversity_donors.json", {"restricted_cities": cities,
            "all_cities": sorted(data["donor_full"].city.unique()), "selection": "SHA256 city-name order, no target outcomes",
            "hyperparameters": "copied unchanged from all-donor selected source/meta; same fixed update counts",
            "interpretation": "donor-pool-size ablation changes both city diversity and unique example volume; no isolated causal diversity claim",
            "restricted_counts": _train_fields(restricted), "all_donor_counts": _train_fields(data["donor_full"])})
        settings["diversity6_finetune"] = copy.deepcopy(settings["matched_gru_finetune"])
        if c["include_meta"]:
            settings["diversity6_maml_sgd"] = copy.deepcopy(settings["maml_gru_sgd"])
        for seed in c["seeds"]:
            for kind in (["pooled", "maml"] if c["include_meta"] else ["pooled"]):
                name, spec = f"refit_diversity6_{kind}_s{seed}", specs["manual_gru"]
                selected = selected_sources["manual_gru"] if kind == "pooled" else meta_selected["maml"]
                preset = {**selected["preset"], "steps": selected["steps"]}
                fit = (fit_supervised(_new_model(spec, seed, device), restricted, preset, seed=seed, c=c, device=device,
                    checkpoint=out / "checkpoints" / f"{name}.pt") if kind == "pooled" else
                    fit_meta("maml", spec, restricted, preset, seed=seed, c=c, device=device,
                    checkpoint=out / "checkpoints" / f"{name}.pt"))
                _record_fit(out, name, fit, restricted, stage="donor_diversity_fixed_refit", method=f"diversity6_{kind}",
                    seed=seed, preset=preset, spec=spec, extra={"source_donor_count": count, "ablation": "donor_pool_size_diversity_and_volume"})
                sources[(f"diversity6_{kind}", seed)] = fit.model.cpu()
    methods = method_inventory(c)
    base._atomic_json(out / "frozen_adaptation_settings.json", settings)
    base._atomic_json(out / "selected_model_specs.json", specs)
    base._atomic_json(out / "source_selections.json", {"supervised": selected_sources, "meta": meta_selected})
    # Only after all settings and code have been frozen do target prediction blocks begin.
    completed, paths = 0, []
    total = len(data["targets"]) * len(c["budgets"]) * len(c["seeds"]) * len(methods)
    for city, parts in sorted(data["targets"].items()):
        queries = [parts["validation"], parts["test"]]
        for budget in c["budgets"]:
            support = base.select_budget(parts["adapt"], budget)
            support_dates = _label_dates(support)
            for seed in c["seeds"]:
                for method in methods:
                    name = f"{base._slug(city)}_b{budget}_s{seed}_{method}"
                    path = out / "prediction_blocks" / f"{name}.csv.gz"
                    record = out / "fit_records" / f"target_{name}.json"
                    paths.append(path)
                    if path.exists() and record.exists() and c["resume"]:
                        completed += 1
                        continue
                    _check_budget(c)
                    print(f"[research] target {city} B={budget} seed={seed} {method}", flush=True)
                    began, fitted, model, preset, spec = time.perf_counter(), None, None, None, None
                    if method in BASELINES:
                        predictions, params, _, seconds = base._baseline(method, support, queries, seed, c, settings.get(method))
                    else:
                        if method.startswith("deep_"):
                            _, architecture, mode = method.split("_", 2)
                            spec = specs[architecture]
                            model = (_new_model(spec, seed, device) if mode == "local" else copy.deepcopy(sources[(architecture, seed)]).to(device))
                        elif method.startswith("matched_gru_"):
                            mode, spec = method.removeprefix("matched_gru_"), specs["manual_gru"]
                            model = (_new_model(spec, seed, device) if mode == "local" else copy.deepcopy(sources[("manual_gru", seed)]).to(device))
                        elif method.startswith("diversity6_"):
                            mode, spec = method.removeprefix("diversity6_"), specs["manual_gru"]
                            kind = "maml" if mode == "maml_sgd" else "pooled"
                            model = copy.deepcopy(sources[(f"diversity6_{kind}", seed)]).to(device)
                        else:
                            algorithm, _, optimizer = method.split("_")
                            mode, spec = optimizer, specs["manual_gru"]
                            model = copy.deepcopy(sources[(algorithm, seed)]).to(device)
                        preset = settings[method] if method in settings else None
                        if preset is not None:
                            fitted = fit_supervised(model, support, preset, seed=seed, c=c, device=device, adaptation=True,
                                checkpoint=out / "checkpoints" / f"target_{name}.pt")
                            model = fitted.model
                        predictions = [base.predict_neural(model, q, device) for q in queries]
                        params, seconds = sum(p.numel() for p in model.parameters()), fitted.elapsed if fitted else 0.0
                    block = pd.concat([base._prediction_rows(q, p, method=method, budget=budget, seed=seed, period=period)
                        for q, p, period in zip(queries, predictions, ("secondary_2024", "test_2025"))], ignore_index=True)
                    trained = method not in ("persistence", "seasonal7", "mean7") and not method.endswith("pooled_zero")
                    row = {"stage": "target_fit", "method": method, "seed": seed, "city": city, "budget": budget,
                        "architecture": model.architecture if model is not None else method,
                        "model_spec": spec, "parameters": params, **_train_fields(support),
                        "train_examples": len(support) if trained else 0, "available_budget_examples": len(support),
                        "support_label_dates": support_dates, "support_label_dates_sha256": _dates_digest(support_dates),
                        "optimizer": preset["optimizer"] if preset else None, "lr": preset["lr"] if preset else None,
                        "requested_steps": int(preset["steps"]) if preset else 0,
                        "selected_step": int(preset["steps"]) if preset else 0,
                        "actual_steps": fitted.actual_steps if fitted else 0,
                        "fit_seconds": seconds, "block_seconds": time.perf_counter() - began,
                        "selection_data": "donor-only settings frozen before target fits", "prediction_examples": len(block),
                        "baseline_preset": settings.get(method) if method in BASELINES else None,
                        "source_key": _source_key(method), "adaptation_preset": preset}
                    if model is not None:
                        row.update(_cost(model))
                    base._atomic_json(record, row)
                    if c["save_target_states"] and model is not None:
                        base._torch_save(out / "target_states" / f"{name}.pt", {"state_dict": base._state(model),
                            "fingerprint": fingerprint, "method": method, "city": city, "budget": budget,
                            "seed": seed, "model_spec": spec, "preset": preset})
                    base._atomic_csv(path, block)
                    # Result and record are committed: target optimizer state is now
                    # redundant because resume skips this exact complete block.
                    _prune([out / "checkpoints" / f"target_{name}.pt"])
                    completed += 1
                    base._atomic_json(out / "progress.json", {"stage": "target_fits", "completed_blocks": completed,
                        "total_blocks": total, "last_completed": name, "methods": methods,
                        "elapsed_seconds_this_invocation": time.perf_counter() - started})
    predictions = pd.concat([pd.read_csv(path, dtype={"station_id": str}) for path in paths], ignore_index=True)
    fitlog = pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / "fit_records").glob("*.json"))])
    expected = sum(len(p["validation"]) + len(p["test"]) for p in data["targets"].values()) * len(c["budgets"]) * len(c["seeds"]) * len(methods)
    if len(predictions) != expected or predictions.duplicated(["city", "period", "date", "horizon", "budget", "seed", "method"]).any():
        raise AssertionError("Research prediction coverage/unique-key failure")
    if set(predictions.method) != set(methods):
        raise AssertionError("Frozen method coverage failed")
    base._atomic_csv(out / "predictions.csv.gz", predictions)
    base._atomic_csv(out / "fit_log.csv", fitlog)
    base._atomic_csv(out / "source_training_diagnostics.csv", fitlog[fitlog.stage.str.startswith("donor_")])
    base._atomic_json(out / "compute_summary.json", {"fit_seconds_sum": float(fitlog.fit_seconds.fillna(0).sum()),
        "invocation_wall_seconds": time.perf_counter() - started, "device": str(device),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "timing_note": "Fit elapsed includes donor validation/checkpoint overhead and prior resumed fit time; source HPO development calibrations have separate checkpoints"})
    base._atomic_json(out / "progress.json", {"stage": "complete", "completed_blocks": completed,
        "total_blocks": total, "prediction_rows": len(predictions), "fit_rows": len(fitlog),
        "methods": methods, "fingerprint": fingerprint, "elapsed_seconds_this_invocation": time.perf_counter() - started})
    failure = out / "failure.json"
    if failure.exists():
        failure.unlink()
    return predictions, fitlog


def _source_key(method: str) -> str | None:
    if method in BASELINES or method.endswith("_local"):
        return None
    if method.startswith("deep_"):
        return method.split("_")[1]
    if method.startswith("matched_gru_"):
        return "manual_gru"
    if method.startswith("diversity6_"):
        return "diversity6_maml" if method.endswith("maml_sgd") else "diversity6_pooled"
    return method.split("_")[0]


def gradient_smoke(device: torch.device | None = None) -> dict[str, Any]:
    """Artificial-data proof that full MAML retains nonzero Hessian contributions."""
    device = device or torch.device("cpu")
    base._seed(551)
    model = ManualGRUForecaster(8, 2).to(device)
    x = torch.randn(24, 7, 6, device=device) * 0.4
    y = 0.5 * x[:, -1, 0] + 0.2 * x[:, 0, 1] - 0.1
    batches = [np.arange(0, 10), np.arange(2, 12), np.arange(1, 11)]
    query, preset = np.arange(12, 24), {"inner_lr": 0.1}
    full, full_loss = _meta_gradient(model, x, y, batches, query, preset, "maml", 100.0)
    first, first_loss = _meta_gradient(model, x, y, batches, query, preset, "fomaml", 100.0)
    difference = math.sqrt(sum(float((a - b).square().sum().cpu()) for a, b in zip(full, first)))
    magnitude = math.sqrt(sum(float(a.square().sum().cpu()) for a in full))
    body_grad = math.sqrt(sum(float(g.square().sum().cpu()) for (name, _), g in zip(model.named_parameters(), full) if name.startswith("layers.")))
    before = base._state(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter, gradient in zip(model.parameters(), full):
        parameter.grad = gradient
    opt.step()
    after = base._state(model)
    body_delta = sum(float((after[k] - v).abs().sum()) for k, v in before.items() if k.startswith("layers."))
    if difference <= 1e-7 or magnitude <= 1e-7 or body_grad <= 1e-8 or body_delta <= 1e-8:
        raise AssertionError("Second-order MAML gradient/body update proof failed")
    if abs(full_loss - first_loss) > 1e-6:
        raise AssertionError("MAML/FOMAML smoke must compare identical forward adaptation")
    # Manual gate implementation equivalence with a native one-layer GRU.
    native = nn.GRU(6, 8, batch_first=True).to(device)
    manual = ManualGRULayer(6, 8).to(device)
    with torch.no_grad():
        manual.input_gates.weight.copy_(native.weight_ih_l0)
        manual.input_gates.bias.copy_(native.bias_ih_l0)
        manual.hidden_gates.weight.copy_(native.weight_hh_l0)
        manual.hidden_gates.bias.copy_(native.bias_hh_l0)
    equivalence = float((manual(x) - native(x)[0]).abs().max().detach().cpu())
    if equivalence > 1e-5:
        raise AssertionError("Manual GRU gates differ from native GRU equations")
    return {"passed": True, "data": "artificial tensors only", "device": str(device),
        "full_maml_vs_fomaml_gradient_l2": difference, "full_gradient_l2": magnitude,
        "body_gradient_l2": body_grad, "body_parameter_absolute_change": body_delta,
        "forward_query_loss_maml": full_loss, "forward_query_loss_fomaml": first_loss,
        "manual_native_gru_max_abs_difference": equivalence, "create_graph_full_maml": True,
        "create_graph_fomaml": False, "cudnn_used_for_meta": False}


def smoke_test(out: str | Path) -> dict[str, Any]:
    """Tiny artificial run covers all methods, resume, source selection and purges."""
    donors = ("donor_a", "donor_b", "donor_dev")
    train = pd.concat([base._synthetic_frame(city, "2022-01-01", "2022-12-31", i) for i, city in enumerate(donors)], ignore_index=True)
    val = pd.concat([base._synthetic_frame(city, "2023-01-04", "2023-01-18", i) for i, city in enumerate(donors)], ignore_index=True)
    target = {"adapt": base._synthetic_frame("target", "2023-01-01", "2023-12-31", 5),
        "validation": base._synthetic_frame("target", "2024-01-04", "2024-01-08", 5),
        "test": base._synthetic_frame("target", "2025-01-04", "2025-01-08", 5)}
    bundle = {"donor_train": train, "donor_val": val, "donor_full": pd.concat([train, val], ignore_index=True),
        "targets": {"target": target}, "dev_cities": ["donor_dev"]}
    c = {"smoke": True, "seeds": [17], "budgets": [5], "pseudo_budgets": [5],
        "manual_hidden": 8, "manual_layers": 2, "meta_validation_budget": 5,
        "source_grid": [{"hidden": 16, "lr": 0.001, "weight_decay": 0.0001, "steps": 2}],
        "meta_grid": [{"inner_lr": 0.03, "outer_lr": 0.001, "inner_steps": 3, "steps": 2}],
        "source_eval_every": 1, "source_min_steps": 1, "meta_eval_every": 1, "meta_min_steps": 1,
        "meta_tasks": 1, "meta_query_days": 3, "source_hpo_adaptation_steps": 2,
        "source_batch_size": 12, "meta_batch_size": 8, "diversity_donors": 1,
        "deep_scratch_grid": [{"optimizer": "adam", "lr": 0.001, "steps": 2, "weight_decay": 0.0}],
        "deep_finetune_grid": [{"optimizer": "adam", "lr": 0.001, "steps": 2, "weight_decay": 0.0}],
        "native_sgd_grid": [{"optimizer": "sgd", "lr": 0.03, "steps": 3, "weight_decay": 0.0}],
        "meta_adam_grid": [{"optimizer": "adam", "lr": 0.001, "steps": 2, "weight_decay": 0.0}],
        "histgb_iterations": 3, "rf_trees": 3, "compute_budget_seconds": 600, "device": "cpu"}
    predictions, logs = train_research(bundle, out, c)
    again, _ = train_research(bundle, out, c)
    pd.testing.assert_frame_equal(predictions, again)
    expected = len(target["validation"]) + len(target["test"])
    assert len(predictions) == expected * len(METHODS)
    assert set(predictions.method) == set(METHODS)
    for _, group in predictions.groupby("method"):
        assert len(group) == expected
    result = {"passed": True, "methods": len(METHODS), "prediction_rows": len(predictions),
        "fit_rows": len(logs), "resume_identical": True, "gradient": gradient_smoke()}
    base._atomic_json(Path(out) / "smoke_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gradient-smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp/aqi-research-smoke"))
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(base._jsonable(smoke_test(args.out)), indent=2))
    elif args.gradient_smoke:
        print(json.dumps(gradient_smoke(), indent=2))
    else:
        parser.error("Use train_research(bundle,outdir,config), --smoke, or --gradient-smoke")
