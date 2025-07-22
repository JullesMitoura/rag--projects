import logging

# predefined log formats
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s' # Não usado nesta versão simples, mas mantido
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logger(name: str = __name__) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger