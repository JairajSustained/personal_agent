#!/usr/bin/env python3
"""Personal Agent - A multi-provider chat agent with GUI."""

import logging

from gui import main


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

if __name__ == "__main__":
    configure_logging()
    main()
