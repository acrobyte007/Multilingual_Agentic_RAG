import logging
import traceback


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_info(logger: logging.Logger, message: str):
    logger.info(message)


def log_error(logger: logging.Logger, message: str):
    logger.error(message)


def log_exception(logger: logging.Logger, error: Exception):
    tb = traceback.format_exc()
    logger.error(f"{error}\n{tb}")