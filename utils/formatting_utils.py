import logging


COLOR_DEFAULT = '\x1b[38;5;250m'
COLOR_INFO = '\x1b[38;5;255m'
COLOR_DEBUG = '\x1b[38;5;245m'
COLOR_WARN = '\x1b[33;20m'
COLOR_FATAL = '\x1b[31;20m'
COLOR_RESET = '\x1b[0m'


class Formatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        formats = COLOR_DEFAULT + '%(asctime)s - %(levelname)s: '
        if record.levelno == logging.DEBUG:
            formats += COLOR_DEBUG
        elif record.levelno == logging.WARN:
            formats += COLOR_WARN
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            formats += COLOR_FATAL
        else:
            formats += COLOR_INFO
        formats += '%(message)s' + COLOR_RESET
        default_formatter = logging.Formatter(formats, datefmt='%H:%M:%S')
        return default_formatter.format(record)


def setup_formatter(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter())

    logging.basicConfig(level=logging.DEBUG, handlers=[handler])
