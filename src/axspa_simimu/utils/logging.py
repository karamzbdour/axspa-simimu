"""
Weights & Biases logging utility and experiment tracking abstraction for axSpA-SimIMU.
"""

import os
from typing import Any, Dict, List, Optional, Union
import logging

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Wrapper for Weights & Biases (wandb) experiment logging.
    Handles metric logging, figures/waveforms, summary tables, and model artifact versioning.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        project: str = "axspa-simimu",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online", # "online", "offline", or "disabled"
        enabled: bool = True,
    ):
        self.enabled = enabled and HAS_WANDB and mode != "disabled"
        self.run = None

        if not self.enabled:
            if not HAS_WANDB and enabled:
                logger.warning("wandb is not installed. Experiment tracking is disabled. Run `pip install wandb`.")
            else:
                logger.info("WandB logging is disabled.")
            return

        # Convert OmegaConf config to standard dict if needed
        config_dict = {}
        if config is not None:
            if hasattr(config, "__dict__"):
                config_dict = vars(config)
            elif hasattr(config, "items"):
                try:
                    from omegaconf import OmegaConf
                    config_dict = OmegaConf.to_container(config, resolve=True)
                except Exception:
                    config_dict = dict(config)
            else:
                config_dict = {"config": str(config)}

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=job_type,
            tags=tags or ["axspa", "virtual-imu"],
            notes=notes,
            config=config_dict,
            mode=mode,
            reinit=True,
        )
        logger.info(f"Initialized Weights & Biases Run: {wandb.run.name} ({wandb.run.url})")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log scalar metrics (loss, MPJPE, ROM violation rate, learning rate).
        """
        if not self.enabled or self.run is None:
            return

        formatted = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            if hasattr(v, "item"):
                formatted[key] = v.item()
            else:
                formatted[key] = float(v)

        if epoch is not None:
            formatted["epoch"] = epoch

        wandb.log(formatted, step=step)

    def log_figure(self, tag: str, figure: Any, step: Optional[int] = None) -> None:
        """
        Log a matplotlib figure (e.g. 3D pose skeleton render or IMU acceleration waveform).
        """
        if not self.enabled or self.run is None:
            return

        wandb.log({tag: wandb.Image(figure)}, step=step)

    def log_table(self, tag: str, columns: List[str], data: List[List[Any]], step: Optional[int] = None) -> None:
        """
        Log a structured evaluation table (e.g. per-action MPJPE / BASMI breakdown).
        """
        if not self.enabled or self.run is None:
            return

        table = wandb.Table(columns=columns, data=data)
        wandb.log({tag: table}, step=step)

    def log_artifact(
        self,
        file_or_dir_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log and version a model checkpoint or synthetic dataset artifact.
        """
        if not self.enabled or self.run is None or not os.path.exists(file_or_dir_path):
            return

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata or {},
        )
        if os.path.isdir(file_or_dir_path):
            artifact.add_dir(file_or_dir_path)
        else:
            artifact.add_file(file_or_dir_path)

        self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        logger.info(f"Logged artifact {artifact_name} to WandB.")

    def finish(self) -> None:
        """Close WandB run cleanly."""
        if self.enabled and self.run is not None:
            wandb.finish()


def init_logger(cfg: Any) -> WandbLogger:
    """
    Helper function to initialize WandbLogger directly from an OmegaConf / Hydra config object.
    """
    logging_cfg = getattr(cfg, "logging", {})
    use_wandb = getattr(logging_cfg, "use_wandb", True)
    project = getattr(logging_cfg, "project", "axspa-simimu")
    entity = getattr(logging_cfg, "entity", None)
    mode = getattr(logging_cfg, "mode", "online")
    group = getattr(logging_cfg, "group", None)
    job_type = getattr(logging_cfg, "job_type", "train")
    tags = list(getattr(logging_cfg, "tags", ["axspa", "virtual-imu"]))
    notes = getattr(logging_cfg, "notes", None)
    name = getattr(cfg, "exp_name", None)

    return WandbLogger(
        config=cfg,
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        tags=tags,
        notes=notes,
        mode=mode,
        enabled=use_wandb,
    )
