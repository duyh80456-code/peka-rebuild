"""Save resolved configs as YAML for run provenance.

Configs are NEVER read back from these YAMLs — they are write-only artifacts
that record what was run.
"""
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml
from hydra_zen import to_yaml


def save_yaml_snapshot(output_dir: os.PathLike, configs: Mapping[str, Any]) -> Path:
    """Write each config to `<output_dir>/configs/<name>.yaml`.

    Args:
        output_dir: experiment output directory
        configs: dict mapping config name → hydra-zen config OR dataclass OR dict
    """
    out = Path(output_dir) / "configs"
    out.mkdir(parents=True, exist_ok=True)

    for name, cfg in configs.items():
        path = out / f"{name}.yaml"
        try:
            text = to_yaml(cfg)
        except Exception:
            if is_dataclass(cfg):
                text = yaml.safe_dump(asdict(cfg), default_flow_style=False)
            elif isinstance(cfg, Mapping):
                text = yaml.safe_dump(dict(cfg), default_flow_style=False)
            else:
                text = repr(cfg)
        path.write_text(text)
    return out
