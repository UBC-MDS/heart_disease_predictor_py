import logging


def setup_logger(script_name):
    """
    Set up and return a logger for the given script.

    Parameters
    ----------
    script_name: str
        The name of the script to use as the logger name.

    Returns
    -------
    logger: logging.Logger
    """
    logger = logging.getLogger(script_name)

    logging.basicConfig(
        filename='ERROR.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
        filemode='a'
    )

    return logger
