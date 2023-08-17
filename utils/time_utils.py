import time
import logging

def functimer(func):
    def wrapper(*args, **kwargs):
        logging.info("start")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.3f} seconds")
        return result
    return wrapper