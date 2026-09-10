"""Уведомления о ходе обучения в мессенджер.

Прогон идёт часами на удалённой машине, и без уведомлений единственный способ
узнать, что он упал на второй эпохе, — зайти и посмотреть. HTTP живёт здесь,
чтобы движок про него не знал.

Два правила, из которых всё остальное следует:

* **Без токена — тишина, а не ошибка.** `build_notifier` возвращает `Notifier`,
  у которого все методы пустые. Эксперимент не должен падать из-за того, что
  мессенджер недоступен или переменные окружения не заданы.
* **Токен не попадает в вывод.** Он есть в URL запроса, поэтому текст любой
  ошибки перед печатью проходит через `_scrub`: иначе токен осел бы в логе
  прогона, который потом уходит в репозиторий или в чужие руки.

Токен и чат берутся из `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` в окружении или
в `.env` (он в `.gitignore`) — тем же способом, что и пути в `src/config.py`.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import dotenv_values

import global_config
from src.progress import ConsoleProgress

MESSAGE_LIMIT = 4096  # предел Telegram на одно сообщение
API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Пустой приёмник: интерфейс без побочных эффектов.

    Движок вызывает его безусловно, поэтому «уведомления выключены» — это
    объект, а не проверка на `None` в четырёх местах вызова.
    """

    enabled = False

    def send(self, text: str) -> bool:
        return False

    def epoch(self, run_name: str, epoch: int, epochs: int, metrics: Mapping[str, float],
              *, best_aic: float) -> bool:
        return self.send(format_epoch(run_name, epoch, epochs, metrics, best_aic=best_aic))

    def finished(self, run_name: str, summary: Mapping[str, object], *, elapsed: float) -> bool:
        return self.send(format_finished(run_name, summary, elapsed=elapsed))

    def failed(self, run_name: str, error: BaseException) -> bool:
        return self.send(f"{run_name}\nПРОГОН УПАЛ\n\n{type(error).__name__}: {error}")


@dataclass(frozen=True)
class TelegramNotifier(Notifier):
    token: str
    chat_id: str
    timeout: float = 10.0

    enabled = True

    def send(self, text: str) -> bool:
        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text[:MESSAGE_LIMIT],
            "disable_web_page_preview": "true",
        }).encode()
        request = urllib.request.Request(API_URL.format(token=self.token), data=payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except Exception as error:  # noqa: BLE001 — мессенджер не должен ронять обучение
            detail = self._scrub(f"{type(error).__name__}: {error}")
            ConsoleProgress.info(f"Telegram: сообщение не отправлено — {detail}")
            return False

    def _scrub(self, text: str) -> str:
        return text.replace(self.token, "***")


def build_notifier(env: Mapping[str, str] | None = None) -> Notifier:
    """Уведомления из окружения; без токена или чата — пустой приёмник."""
    if env is None:
        env = {**dotenv_values(global_config.PROJECT_ROOT / ".env"), **os.environ}
    token = (env.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (env.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return Notifier()
    return TelegramNotifier(token=token, chat_id=chat_id)


def _number(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _terms(metrics: Mapping[str, float], prefix: str) -> str:
    """Слагаемые цели одной строкой: `bce 0.1204  dice 0.2011  cls 0.0631`."""
    terms = {key[len(prefix):]: value for key, value in metrics.items()
             if key.startswith(prefix) and key != f"{prefix}main"}
    return "  ".join(f"{name} {_number(value)}" for name, value in sorted(terms.items()))


def format_epoch(run_name: str, epoch: int, epochs: int, metrics: Mapping[str, float],
                 *, best_aic: float) -> str:
    """Одна эпоха: слагаемые цели и составляющие метрики.

    Именно составляющие, а не только AIC: арм может выиграть Dice и проиграть
    FPR, и по одному числу этого не видно.
    """
    aic = metrics.get("val/aic_tuned", float("nan"))
    lines = [run_name, f"эпоха {epoch}/{epochs} · {ConsoleProgress.format_duration(metrics.get('epoch_time_s', 0))}"]

    full_frame = metrics.get("train/full_frame_p")
    if full_frame is not None:
        lines[-1] += ("  ·  только целые кадры" if full_frame >= 1.0
                      else f"  ·  {1 - full_frame:.0%} кропов")

    lines += ["", f"train loss {_number(metrics.get('train/loss'))}"]
    if train_terms := _terms(metrics, "train/loss_"):
        lines.append(f"  {train_terms}")
    if val_terms := _terms(metrics, "val/loss_"):
        lines += [f"val loss   {_number(metrics.get('val/loss_main'))}", f"  {val_terms}"]

    record = "  ← рекорд" if aic > best_aic else f"  (лучший {_number(best_aic)})"
    lines += ["", f"AIC {_number(aic)}{record}",
              f"  dice_pos {_number(metrics.get('val/dice_tuned'))}"
              f"  fpr_neg {_number(metrics.get('val/fpr_tuned'))}",
              f"  порог маски {_number(metrics.get('val/best_thr'))}"
              f"  порог cls {_number(metrics.get('val/best_cls_thr'))}",
              "", f"lr {_number(metrics.get('train/lr'))}"
                  f"  показов {metrics.get('samples')}"
                  f"  GPU {metrics.get('gpu_gb')} ГБ"]
    if metrics.get("train/skipped_steps"):
        lines.append(f"  пропущено шагов {metrics['train/skipped_steps']}")
    return "\n".join(lines)


def format_finished(run_name: str, summary: Mapping[str, object], *, elapsed: float) -> str:
    best = summary.get("best") or {}
    lines = [run_name, f"ОБУЧЕНИЕ ЗАВЕРШЕНО · {ConsoleProgress.format_duration(elapsed)}", "",
             f"лучший AIC {_number(summary.get('best_aic'))}"]
    if isinstance(best, Mapping) and best:
        lines += [f"  dice_pos {_number(best.get('dice_pos'))}  fpr_neg {_number(best.get('fpr_neg'))}",
                  f"  порог маски {_number(best.get('mask_threshold'))}"
                  f"  порог cls {_number(best.get('cls_threshold'))}"]
    lines.append(f"  показов {summary.get('samples')}")
    return "\n".join(lines)


__all__ = ["Notifier", "TelegramNotifier", "build_notifier", "format_epoch", "format_finished"]
