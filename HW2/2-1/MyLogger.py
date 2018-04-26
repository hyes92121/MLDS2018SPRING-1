import os 
import sys
import logging
import datetime
from functools import wraps

if not os.path.exists('logs'):
    os.mkdir('logs')

# define a global logger for all files. This is the parent of all loggers in distinct files.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# configure streamhandler
sh = logging.StreamHandler(sys.stdout)
str_fmt = "%(name)s - %(message)s"
fmter = logging.Formatter(str_fmt)
sh.setFormatter(fmter)

# configure filehandler
fh = logging.FileHandler("logs/{}.log".format(datetime.datetime.now().strftime("%m%d.%H%M")))
strfmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
datefmt = "%Y-%m-%d %H:%M"
fmter = logging.Formatter(strfmt, datefmt)
fh.setFormatter(fmter)

# add handlers to logger
logger.addHandler(sh)
logger.addHandler(fh)

# define decorator functions for logging

def log_function(logger):
    def wrap(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            displayname = "[{}]".format(function.__name__)
            logger.debug("Calling function {} with args = {}, kwargs = {}".format(displayname, args, kwargs))

            try:
                response = function(*args, **kwargs)
            except Exception as error:
                logger.debug("Function '{}' raised {} with error '{}'".format(
                    function.__name__,
                    error.__class__.__name__,
                    str(error)
                ))
                raise error

            logger.debug("Function {} returned {}".format(displayname, response))

            return response
        return wrapper
    return wrap

def log_method(logger):
    def wrap(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            classname = args[0].__class__.__name__
            displayname = "[{}.{}]".format(classname, function.__name__)
            logger.debug("Calling method {}".format(displayname))

            try:
                response = function(*args, **kwargs)
            except Exception as error:
                logger.debug("Method '{}' raised {} with error '{}'".format(
                    function.__name__,
                    error.__class__.__name__,
                    str(error)
                ))
                raise error

            #logger.debug("Method {} returned {}".format(displayname, response))

            return response
        return wrapper
    return wrap

if __name__ == "__main__":
    logger.info("Testing Logger Module")


