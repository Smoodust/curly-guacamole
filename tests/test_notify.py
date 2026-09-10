"""Tests for training notifications (src/notify).

The point of these is the failure modes, not the happy path: a run that dies
because the messenger is down, or a bot token that ends up in a run log, are
both worse than having no notifications at all.
"""

import urllib.error
import urllib.request

import pytest

from src.notify import MESSAGE_LIMIT, Notifier, TelegramNotifier, build_notifier, format_epoch

TOKEN = "123456:AAxxSECRETxx"
ENV = {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": "42"}

METRICS = {
    "train/loss": 0.4213,
    "train/loss_bce": 0.1204,
    "train/loss_dice": 0.2011,
    "train/loss_fpr": 0.0367,
    "val/loss_bce": 0.1101,
    "val/loss_dice": 0.1932,
    "val/loss_main": 0.3033,
    "val/aic_tuned": 0.7421,
    "val/dice_tuned": 0.8012,
    "val/fpr_tuned": 0.0410,
    "val/best_thr": 0.35,
    "val/best_cls_thr": 0.5,
    "train/lr": 0.00024,
    "train/skipped_steps": 0,
    "train/full_frame_p": 0.5,
    "samples": 36000,
    "gpu_gb": 21.3,
    "epoch_time_s": 432.0,
}


class _Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    """Capture the request instead of talking to Telegram."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return _Response(200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_without_a_token_notifications_are_silent_not_broken():
    """A machine with no token configured must still train."""
    for env in ({}, {"TELEGRAM_BOT_TOKEN": TOKEN}, {"TELEGRAM_CHAT_ID": "42"},
                {"TELEGRAM_BOT_TOKEN": "  ", "TELEGRAM_CHAT_ID": "42"}):
        notifier = build_notifier(env)
        assert type(notifier) is Notifier
        assert not notifier.enabled
        assert notifier.epoch("run", 1, 8, METRICS, best_aic=0.0) is False
        assert notifier.failed("run", RuntimeError("boom")) is False


def test_a_configured_environment_sends_to_the_chat(sent):
    assert build_notifier(ENV).send("привет") is True

    request = sent[0]
    assert request.full_url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    body = request.data.decode()
    assert "chat_id=42" in body
    assert "text=%D0%BF%D1%80%D0%B8%D0%B2%D0%B5%D1%82" in body


def test_a_dead_messenger_does_not_kill_the_run(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert build_notifier(ENV).send("привет") is False


def test_the_token_never_reaches_the_console(monkeypatch, capsys):
    """The token lives in the URL, and urllib puts the URL into its errors."""
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    notifier = build_notifier(ENV)
    assert notifier.send("привет") is False

    printed = capsys.readouterr().out
    assert TOKEN not in printed
    assert "401" in printed


def test_an_oversized_message_is_clipped_to_the_api_limit(sent):
    build_notifier(ENV).send("x" * (MESSAGE_LIMIT + 500))

    text = dict(pair.split("=", 1) for pair in sent[0].data.decode().split("&"))["text"]
    assert len(text) == MESSAGE_LIMIT


def test_the_epoch_message_carries_the_loss_terms_and_the_metric_parts():
    """One AIC number hides the trade-off: an arm can win Dice and lose FPR."""
    message = format_epoch("pvt_v2_b2_protocol_loss_l4_aic_surrogate", 3, 8, METRICS, best_aic=0.7300)

    assert "pvt_v2_b2_protocol_loss_l4_aic_surrogate" in message
    assert "эпоха 3/8" in message
    for term in ("bce 0.1204", "dice 0.2011", "fpr 0.0367"):
        assert term in message
    assert "AIC 0.7421" in message
    assert "dice_pos 0.8012" in message
    assert "fpr_neg 0.0410" in message
    assert "рекорд" in message, "beating the best AIC must be visible at a glance"
    assert "50% кропов" in message


def test_the_epoch_message_marks_the_final_full_frame_phase():
    message = format_epoch("run", 7, 8, {**METRICS, "train/full_frame_p": 1.0}, best_aic=0.9)

    assert "только целые кадры" in message
    assert "рекорд" not in message


def test_a_partial_metrics_dict_still_formats():
    """Arms differ in their loss terms, so no key can be assumed present."""
    message = format_epoch("run", 1, 8, {"val/aic_tuned": 0.5}, best_aic=0.0)

    assert "AIC 0.5000" in message
    assert len(message) <= MESSAGE_LIMIT


def test_a_notifier_can_be_handed_to_the_runner_instead_of_the_environment():
    """Tests and callers must be able to pass their own sink."""
    from dataclasses import replace

    from src.config import load_experiment_config
    from src.training.engine import ExperimentRunner

    config = load_experiment_config("configs/loss_l4_aic_surrogate.yaml")
    config = replace(config, train=replace(config.train, device="cpu"))
    quiet = Notifier()

    assert ExperimentRunner(config, notifier=quiet).notifier is quiet
    assert isinstance(TelegramNotifier(token=TOKEN, chat_id="42"), Notifier)
