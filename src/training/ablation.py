"""Серия прогонов, отличающихся одним параметром, и таблица их итогов.

Ноутбук должен только выбрать конфигурации и посмотреть на результат, поэтому
порядок запуска, устойчивость к падению одного арма и сборка сравнения живут
здесь и покрыты тестами.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ExperimentConfig, load_experiment_config
from src.progress import ConsoleProgress
from src.training.engine import run_experiment
from src.training.runs import Run


@dataclass(frozen=True)
class ArmResult:
    """Итог одного арма: либо метрики, либо причина, по которой их нет."""

    name: str
    config: ExperimentConfig
    summary: dict[str, Any]
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.error is None

    def as_row(self) -> dict[str, Any]:
        best = self.summary.get("best") or {}
        return {
            "arm": self.name,
            "loss": self.config.loss.name,
            "run_name": self.config.paths.run_name,
            "aic": self.summary.get("best_aic"),
            "dice_pos": best.get("dice_pos"),
            "fpr_neg": best.get("fpr_neg"),
            "mask_threshold": best.get("mask_threshold"),
            "cls_threshold": best.get("cls_threshold"),
            "min_area": best.get("min_area"),
            "samples": self.summary.get("samples"),
            "gflops": self.summary.get("gflops"),
            "error": self.error,
        }


class LossAblation:
    """Прогоняет список конфигураций подряд и складывает итоги в одну таблицу.

    Арм, упавший на своём конфиге, не должен уносить с собой остальные: серия
    идёт дальше, а причина остаётся в таблице. Уже посчитанные прогоны читаются
    с диска, поэтому серию можно продолжить после перезапуска ядра.
    """

    def __init__(self, config_paths: Sequence[str | Path], *,
                 data_path: str | Path | None = None) -> None:
        if not config_paths:
            raise ValueError("config_paths must not be empty")
        self.configs = [self._load(path, data_path) for path in config_paths]
        self._check_unique_run_names()

    @staticmethod
    def _load(path: str | Path, data_path: str | Path | None) -> ExperimentConfig:
        config = load_experiment_config(path)
        if data_path is None:
            return config
        return replace(config, paths=replace(config.paths, data_path=Path(data_path).resolve()))

    def _check_unique_run_names(self) -> None:
        names = [config.paths.run_name for config in self.configs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"arms share a run_name and would overwrite each other: {duplicates}")

    @property
    def arm_names(self) -> list[str]:
        return [self._arm_name(config) for config in self.configs]

    @staticmethod
    def _arm_name(config: ExperimentConfig) -> str:
        return config.paths.run_name.split("_loss_")[-1]

    def run(self, *, skip_completed: bool = True) -> list[ArmResult]:
        results: list[ArmResult] = []
        for index, config in enumerate(self.configs, start=1):
            name = self._arm_name(config)
            done = self._finished_summary(config) if skip_completed else None
            if done is not None:
                ConsoleProgress.info(f"[{index}/{len(self.configs)}] {name}: уже посчитан, пропуск")
                results.append(ArmResult(name, config, done))
                continue

            ConsoleProgress.info(f"[{index}/{len(self.configs)}] {name}: запуск ({config.loss.name})")
            try:
                results.append(ArmResult(name, config, run_experiment(config).summary))
            except Exception as error:  # noqa: BLE001 — арм падает, серия продолжается
                ConsoleProgress.info(f"[{index}/{len(self.configs)}] {name}: упал — {error}")
                traceback.print_exc()
                results.append(ArmResult(name, config, {}, error=f"{type(error).__name__}: {error}"))
        return results

    def _finished_summary(self, config: ExperimentConfig) -> dict[str, Any] | None:
        """Сводка арма, доведённого до конца, иначе None."""
        run_dir = config.paths.runs_path / config.paths.run_name
        if not (run_dir / "summary.json").exists():
            return None
        summary = Run.open(run_dir).summary
        if summary.get("best_aic") is None or summary.get("samples") is None:
            return None
        return summary

    @staticmethod
    def compare(results: Iterable[ArmResult]) -> pd.DataFrame:
        """Таблица по армам: AIC и его составляющие, лучший арм сверху."""
        table = pd.DataFrame([result.as_row() for result in results])
        if table.empty:
            return table
        control = table[table["arm"].str.contains("baseline")]["aic"]
        if not control.empty and pd.notna(control.iloc[0]):
            table["delta_vs_control"] = table["aic"] - control.iloc[0]
        return table.sort_values("aic", ascending=False, na_position="last").reset_index(drop=True)


__all__ = ["ArmResult", "LossAblation"]
