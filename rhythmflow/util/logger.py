import logging
# from datetime import datetime


def get_logger(level=1, to_file=True):
    logger = logging.getLogger('rhythmflow')
    # ts = round(datetime.timestamp(datetime.now()))
    logging_levels = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.ERROR
    }
    # if to_file:
    #     log_name = f'/home/max/repos/rhythmflow/logs/logging/{ts}.log'
    #     logging.basicConfig(
    #         filename=log_name,
    #         filemode='w',
    #         format='%(asctime)s - %(message)s',
    #         level=logging_levels[level]
    #     )
    #     logger.info(f'logs written to {log_name}')
    # else:
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging_levels[level]
    )
    return logger
