from utils import set_logger
import logging

logger = set_logger()
logger.info('aaaaa')
logger.debug('bbbb')


# In another file case
logger = logging.getLogger(__name__)
