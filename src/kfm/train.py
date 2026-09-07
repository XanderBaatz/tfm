import os
import warnings

import hydra
import lightning as L
from dotenv import load_dotenv
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger as LitLogger
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=SyntaxWarning, module="flow_matching")
# settings = Settings()
load_dotenv(".env")

from src.kfm.utils import utils
from tools.logger import Logger

log = Logger(name=__name__, credentials=None)


def train(cfg: DictConfig) -> tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating DataModule <{cfg.data_module._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data_module)

    log.info(f"Instantiating LightningModule <{cfg.lightning_module}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.lightning_module)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[LitLogger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training...")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing...")
        ckpt_path = cfg.get("ckpt_path")
        if not ckpt_path:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if not ckpt_path:
                log.warning("Best checkpoint not found! Using current weights for testing...")
                ckpt_path = None

        log.info(f"Using checkpoint path: {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(config_path="configs", config_name="csp", version_base=None)
def main(cfg: DictConfig) -> float | None:
    if "OUTPUT_DIR" in os.environ:
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir = os.environ["OUTPUT_DIR"]

    # train the model
    metric_dict, _ = train(cfg)

    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric and optimized_metric in metric_dict:
        return metric_dict[optimized_metric].item()

    return None


if __name__ == "__main__":
    main()
