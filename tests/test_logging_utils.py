"""
Unit tests for logging_utils.py: logs go to a file, repeated setup does not duplicate output, arguments are recorded.
logging_utils.py 的单元测试：日志同时写入文件、重复调用不重复输出、记录参数。
"""

import argparse
import logging
import sys

import pytest

from dra.logging_utils import log_args, setup_logging


@pytest.fixture(autouse=True)
def restore_logging():
    """
    setup_logging changes the global root logger and sys.excepthook; restore them after each test.
    setup_logging 会改动全局的根 logger 和 sys.excepthook，测试后恢复原状。
    """
    root = logging.getLogger()
    old_handlers, old_level, old_hook = root.handlers[:], root.level, sys.excepthook
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in old_handlers:
        root.addHandler(handler)
    root.setLevel(old_level)
    sys.excepthook = old_hook


def read_single_log(log_dir):
    [log_file] = log_dir.glob("*.log")
    return log_file.name, log_file.read_text(encoding="utf-8")


def test_messages_go_to_timestamped_file(tmp_path):
    logger = setup_logging("03_retrieve", tmp_path)
    logger.info("R@3: %.2f", 0.93)
    # Modules in src/dra log to the same file / src/dra 里的模块也写进同一文件
    logging.getLogger("dra.data").info("from a library module")

    name, text = read_single_log(tmp_path)
    assert name.startswith("03_retrieve_") and name.endswith(".log")
    assert "INFO 03_retrieve: R@3: 0.93" in text
    assert "INFO dra.data: from a library module" in text


def test_calling_twice_does_not_duplicate_handlers(tmp_path):
    setup_logging("a", tmp_path)
    setup_logging("b", tmp_path)
    assert len(logging.getLogger().handlers) == 2  # terminal + file, one each / 终端 + 文件，各一个


def test_log_args_records_every_argument(tmp_path):
    logger = setup_logging("05_generate_answers", tmp_path)
    log_args(logger, argparse.Namespace(model="Qwen/Qwen2.5-7B-Instruct", seed=42))

    _, text = read_single_log(tmp_path)
    assert "model = Qwen/Qwen2.5-7B-Instruct" in text
    assert "seed = 42" in text


def test_uncaught_exception_is_logged_with_traceback(tmp_path):
    setup_logging("05_generate_answers", tmp_path)
    try:
        raise RuntimeError("CUDA out of memory")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())

    _, text = read_single_log(tmp_path)
    assert "CRITICAL" in text
    assert "RuntimeError: CUDA out of memory" in text
