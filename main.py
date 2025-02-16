from utils.allocateGPU import *
try:
    allocate_gpu()
    print("Running With GPU")
except :
    print("Running Without GPU")
import parser
import _main
from utils.logger import get_logger

# Get the logger
logger = get_logger()

if __name__ == "__main__":
    args = parser.parse_args()
    logger.info("Starting the execution!")
    logger.info("#" * 64)
    for i in vars(args):
        logger.info(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    logger.info("#" * 64)
    _main.main(args)
