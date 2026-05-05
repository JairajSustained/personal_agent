from __future__ import annotations

from gui.chat_window import ChatWorker, MemoryWorker, TitleWorker


class _SignalRaisesRuntimeError:
    def emit(self, *_args, **_kwargs):
        raise RuntimeError("Signal source has been deleted")


class _FakeSignals:
    def __init__(self):
        self.done = _SignalRaisesRuntimeError()
        self.failed = _SignalRaisesRuntimeError()
        self.chunk = _SignalRaisesRuntimeError()


def test_safe_emit_helpers_swallow_runtime_error() -> None:
    signal = _SignalRaisesRuntimeError()

    ChatWorker._safe_emit(signal, "x")
    TitleWorker._safe_emit(signal, "x")
    MemoryWorker._safe_emit(signal, "x")


def test_memory_worker_run_survives_emit_and_model_errors() -> None:
    class _FailingAgent:
        async def generate_memory_update(self, *_args, **_kwargs):
            raise RuntimeError("content_filter")

    worker = MemoryWorker(_FailingAgent(), "mem", "user", "assistant")
    worker.signals = _FakeSignals()

    worker.run()


def test_title_worker_run_survives_emit_and_model_errors() -> None:
    class _FailingAgent:
        async def generate_title(self, *_args, **_kwargs):
            raise RuntimeError("model_error")

    worker = TitleWorker(_FailingAgent(), "hello")
    worker.signals = _FakeSignals()

    worker.run()
