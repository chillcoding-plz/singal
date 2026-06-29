import json
import shutil
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

CONFIG_FILES = {
    "pre_sort": CONFIG_DIR / "pre_sort_config.json",
    "main_sort": CONFIG_DIR / "main_sort_main5_config.json",
    "fine_sort": CONFIG_DIR / "fine_sort_mht_config.json",
    "recognition": CONFIG_DIR / "signal_recognition_config.json",
    "radar_attribute": CONFIG_DIR / "radar_attribute_config.json",
}


def config_path(section: str) -> Path:
    return CONFIG_FILES[section]


def load_config(section: str) -> dict:
    path = config_path(section)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def nested_config(section: str, key: str) -> dict:
    value = load_config(section).get(key, {})
    return value if isinstance(value, dict) else {}


def apply_mapping_to_object(target: object, values: Mapping[str, Any]) -> list[str]:
    applied = []
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)
            applied.append(str(key))
    return applied


def append_cli_args(command: list[str], values: Mapping[str, Any], supported: set[str]) -> list[str]:
    applied = []
    for key, value in values.items():
        if key not in supported:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            command.append(flag if value else f"--no_{key}")
        else:
            command.extend([flag, str(value)])
        applied.append(str(key))
    return applied


def snapshot_config(section: str, run_dir: Path, filename: str | None = None) -> Path | None:
    source = config_path(section)
    if not source.exists():
        return None
    destination = Path(run_dir) / (filename or source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination
