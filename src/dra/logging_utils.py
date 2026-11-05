"""
Logging setup shared by all scripts.
所有脚本共用的日志设置。

Every message goes both to the terminal and to a timestamped log file, so long runs on the
cluster (where the terminal may be gone) can still be inspected afterwards.
每条信息同时输出到终端和带时间戳的日志文件；在集群上跑长任务时，即使终端已断开，
事后也能查看日志。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name, log_dir, level=logging.INFO):
    """
    Configure logging for one script run and return its logger.
    为一次脚本运行设置日志，返回该脚本的 logger。

    Log file: <log_dir>/<name>_<YYYYmmdd-HHMMSS>.log
    Uncaught exceptions are also written to the log file with their traceback.
    未捕获的异常（含 traceback）也会写进日志文件。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.log"

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Configure the root logger so that loggers in src/dra/ also end up in the same file.
    # Remove old handlers first, so calling this twice does not print every line twice.
    # 配置根 logger，src/dra/ 里各模块的日志也会写进同一个文件。
    # 先移除旧的 handler，避免重复调用时每行输出两遍。
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # huggingface_hub logs every HTTP request at INFO level; keep only its warnings.
    # huggingface_hub 会在 INFO 级别记录每个 HTTP 请求，只保留它的警告。
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(name)

    def log_uncaught_exception(exc_type, exc_value, exc_traceback):
        # Ctrl+C: exit as usual, not logged as an error / Ctrl+C：照常退出，不记为错误
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exception

    logger.info("Log file: %s", log_path)
    return logger


def log_args(logger, args):
    """
    Record every command-line argument, so each log file shows exactly how the run was configured.
    记录全部命令行参数，使每个日志文件都能说明这次运行用的是什么配置（模型、种子、路径等）。
    """
    logger.info("Arguments:")
    for key, value in sorted(vars(args).items()):
        logger.info("  %s = %s", key, value)
