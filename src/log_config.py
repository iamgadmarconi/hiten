import logging
import sys

def setup_logging(level=logging.INFO, format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """Configures basic logging to stdout."""
    logging.basicConfig(
        level=level,
        format=format_string,
        stream=sys.stdout  # Explicitly set stream to stdout
    )

# Setup logging when this module is imported
setup_logging()

# Create a logger instance for other modules to import
logger = logging.getLogger(__name__)
