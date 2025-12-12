import argparse
import json
import os
import sys

import pandas as pd

from src.services.config_generator import ConfigGenerator
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Main entry point for config generation CLI. Returns: None."""
    parser = argparse.ArgumentParser(
        description="Generate salary forecast configuration from data."
    )
    parser.add_argument("input_file", help="Path to input CSV file.")
    parser.add_argument("-o", "--output", help="Output JSON file path (default: stdout).")
    parser.add_argument("--llm", action="store_true", help="Use LLM to infer configuration.")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider (default: openai).",
    )
    parser.add_argument("--preset", default="none", help="LLM Preset (e.g. salary, none).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    log_level = "INFO" if args.verbose else "WARNING"
    setup_logging(level=log_level)

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded data with shape {df.shape}")

        generator = ConfigGenerator()
        logger.info(f"Generating config (LLM={args.llm})...")
        config = generator.generate_config(
            df, use_llm=args.llm, provider=args.provider, preset=args.preset
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration written to {args.output}")
        else:
            print(json.dumps(config, indent=4))

    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
