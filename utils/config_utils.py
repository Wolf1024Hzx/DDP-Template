"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

import os

import logger
import yaml


def _item_in_config(item: str, config: dict) -> None:
    if item not in config:
        raise RuntimeError(f"{item} not in config file")


def _generate_output_dir(dir_path) -> None:
    os.makedirs(dir_path, exist_ok=True)


def load_config(config_path: str) -> dict:
    default_log_config = {
        "LOG_FILE": None,
        "ONLY_MAIN_PROCESS": True
    }
    default_logger = logger.Logger(default_log_config)
    try:
        default_logger.debug("Loading config...")
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        # environment
        _item_in_config("CUDA_VISIBLE_DEVICES", config)
        # train
        _item_in_config("LR", config)
        _item_in_config("BATCH_SIZE", config)
        _item_in_config("EPOCHS", config)
        _item_in_config("CHECKPOINT_PATH", config)
        # test
        _item_in_config("INFERENCE_MODEL_PATH", config)
        # output
        _item_in_config("OUTPUT_DIR", config)
        _item_in_config("EXP_NAME", config)
        _generate_output_dir(config["OUTPUT_DIR"])
        _generate_output_dir(os.path.join(config["OUTPUT_DIR"], config["EXP_NAME"]))
        config["LOG_FILE"] = os.path.join(config["OUTPUT_DIR"], config["EXP_NAME"], "log.txt")
        config["ONLY_MAIN_PROCESS"] = True
        # DanceTrack
        _item_in_config("DANCE_TRACK_ROOT", config)

        default_logger.success("Loaded config!")
        default_logger.update_config(config)
        default_logger.debug(f"config content: {config}")
        return config
    except FileNotFoundError:
        default_logger.error(f"文件 {config_path} 未找到！")
        exit(1)
    except yaml.YAMLError as e:
        default_logger.error(f"解析 YAML 文件时出错: {e}")
        exit(1)
    except RuntimeError as runtime_error:
        default_logger.error(runtime_error)
        exit(1)
