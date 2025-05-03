import logging


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # filename='app.log',  # Save logs to this file
    # filemode='w'  # 'w' to overwrite, 'a' to append
)



logger = logging.getLogger(__name__)



