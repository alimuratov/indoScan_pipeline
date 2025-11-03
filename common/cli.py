import os, argparse, logging
from typing import Callable, Tuple, Optional
from common.config import load_config, Config

def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to config file")

def add_log_level_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

def parse_args_with_config(build_parser: Callable[[], argparse.ArgumentParser], 
                           defaults_from_cfg: Callable[[Config], dict],
                           argv: Optional[list] = None) -> Tuple[argparse.Namespace, Config]:
    """
    1. Build a parser based on the passed build_parser function
    2. Load the config file based on the --config argument (passed as an argument to the parser)
    3. Construct a dictionary of values from the config file based on the paramters specified in the defaults_from_cfg function
    """
    p = build_parser()
    cfg_path = p.parse_known_args(argv)[0].config
    cfg = load_config(cfg_path)
    p.set_defaults(**defaults_from_cfg(cfg))
    args = p.parse_args()
    return args, cfg

def setup_logging(log_level: str) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')