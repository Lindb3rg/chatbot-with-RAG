import logging


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # filename='app.log',  # Save logs to this file
    # filemode='w'  # 'w' to overwrite, 'a' to append
)



logger = logging.getLogger(__name__)



def log_event(type, message):
    if type == "info":
        logger.info(message)
    elif type == "warning":
        logger.warning(message)
    elif type == "error":
        logger.error(message)
    else:
        logger.warning(f"Log type could not be found for: {message}")
    return


