"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

import datetime
import utils


class Logger:
    def __init__(self, config: dict):
        self.log_file = config['LOG_FILE']
        self.only_main_process = config['ONLY_MAIN_PROCESS']

    def update_config(self, config: dict):
        self.log_file = config['LOG_FILE']
        self.only_main_process = config['ONLY_MAIN_PROCESS']

    def success(self, success: str) -> None:
        success_msg_console = f"[{self._get_datetime()}] {self._color('[SUCCESS]', '\033[92m')} {success}"
        success_msg_file = f"[{self._get_datetime()}] [SUCCESS] {success}"
        self._log_to_file(success_msg_file)
        self._log_to_console(success_msg_console)

    def info(self, info: str) -> None:
        info_msg_console = f"[{self._get_datetime()}] {self._color('[INFO]', '\033[94m')} {info}"
        info_msg_file = f"[{self._get_datetime()}] [INFO] {info}"
        self._log_to_file(info_msg_file)
        self._log_to_console(info_msg_console)

    def debug(self, debug: str) -> None:
        debug_msg_console = f"[{self._get_datetime()}] {self._color('[DEBUG]', '\033[93m')} {debug}"
        debug_msg_file = f"[{self._get_datetime()}] [DEBUG] {debug}"
        self._log_to_file(debug_msg_file)
        self._log_to_console(debug_msg_console)

    def warning(self, warning: str) -> None:
        warning_msg_console = f"[{self._get_datetime()}] {self._color('[WARNING]', '\033[33m')} {warning}"
        warning_msg_file = f"[{self._get_datetime()}] [WARNING] {warning}"
        self._log_to_file(warning_msg_file)
        self._log_to_console(warning_msg_console)

    def error(self, error: str) -> None:
        error_msg_console = f"[{self._get_datetime()}] {self._color('[ERROR]', '\033[91m')} {error}"
        error_msg_file = f"[{self._get_datetime()}] [ERROR] {error}"
        self._log_to_file(error_msg_file)
        self._log_to_console(error_msg_console)

    def _log_to_file(self, msg: str) -> None:
        if self.log_file is None or not self._can_log():
            return
        with open(self.log_file, "a") as f:
            f.writelines(msg + "\n")

    def _log_to_console(self, msg: str) -> None:
        if not self._can_log():
            return
        print(msg)

    def _get_datetime(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _can_log(self) -> bool:
        if self.only_main_process:
            return utils.is_main_process()
        return True

    def _color(self, text: str, color_code: str) -> str:
        return f"{color_code}{text}\033[0m"
