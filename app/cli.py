from __future__ import annotations

import copy
import os

import click
import uvicorn
from uvicorn.config import LOGGING_CONFIG


@click.command()
@click.option("--host", default=None, help="Bind host (overrides config/env)")
@click.option("--port", default=None, type=int, help="Bind port (overrides config/env)")
@click.option("--workers", default=None, type=int, help="Number of workers")
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml file")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode: workers=1, log-level=debug, full tracebacks",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Log level (overrides config/env)",
)
def main(host, port, workers, config, debug, log_level):
    """perplexity-proxy — OpenAI-compatible proxy for Perplexity AI"""

    os.environ["CONFIG_FILE"] = config
    if debug:
        os.environ["DEBUG"] = "true"

    from app.config import settings

    effective_host = host or settings.HOST
    effective_port = port or settings.PORT
    effective_workers = 1 if debug else (workers or settings.WORKERS)
    effective_log_level = "debug" if debug else (log_level or settings.LOG_LEVEL)

    if debug:
        click.echo("Debug mode enabled — workers=1, log-level=debug")

    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["disable_existing_loggers"] = False

    uvicorn.run(
        "app.main:app",
        host=effective_host,
        port=effective_port,
        workers=effective_workers,
        log_level=effective_log_level,
        reload=debug,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
