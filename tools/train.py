import argparse
import os

import wandb
import yaml
from typing import Any, Dict
from ultralytics import YOLO, settings


class Trainer:
    def __init__(self, config_path: str) -> None:
        """
        Initializes the Trainer class with the given configuration file and wandb project name.

        Args:
            config_path (str): Path to the YAML configuration file.
        Raise:
            ValueError: If the configuration file does not exist at the specified path.
        """
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config: Dict[str, Any] = yaml.safe_load(f)
            # wandb logging config for experiment tracking
            self.wandb_logging: Dict[str, Any] = config.get("logging")
            # train config are contain of dynamic hyperparams namely epochs,
            # batch, imgsz, others
            self.train_config: Dict[str, Any] = config.get("train")
            # aug config are contain of static augmentations params namely,
            # hsv_h, hsv_s, hsv_v, others
            self.aug_config: Dict[str, Any] = {}  # config.get('aug')
        else:
            raise ValueError("Config file does not exist")

    def train(self) -> None:
        """
        Executes the training pipeline with wandb logging and model export.

        This method initializes a wandb run, loads and trains a YOLO model,
        exports it to the specified format, and logs the results.

        """
        # inialize wandb run
        # wandb.init(
        #     project=self.wandb_logging.get(
        #         "project",
        #     ),
        #     config=self.train_config,
        #     name=self.wandb_logging.get("run_name", "train"),
        #     job_type='training'
        # )
        settings.update({'wandb':True})

        # train the model
        model = YOLO(self.train_config.get("model", "yolo8n.pt"))
        model.train(
          project=self.wandb_logging.get("project"),
          name=self.wandb_logging.get("run_name", "train"),
          **self.train_config,
          **self.aug_config
          )

        # Export the model into 'ncnn'
        target = self.export(model, "ncnn")

        # export file as the output of wandb
        # wandb.log({"Model is succesfully exported to": target})
        wandb.finish()

    def export(self, model: YOLO, export_format: str= "ncnn", **kwargs) -> str:
        """
        Exports the trained model to a specified format.

        Args:
            model (YOLO): The trained YOLO model instance to export.
            export_format (str, optional): Target export format. Defaults to "ncnn".
        Returns:
            str: Path to the exported model file.
        Raise:
            ValueError: If the model export process fails.
        """
        try:
            target = model.export(format=export_format, **kwargs)
            print("Model Succesfully Exported")
        except ValueError as e:
            raise ValueError("Export failed") from e
        return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="run name")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/settings.yaml",
        help="Path to config path",
    )
    parser.add_argument(
        "--project", type=str, default="cap_detection", help="Wandb project name"
    )
    args = parser.parse_args()
    trainer = Trainer(args.config_path, args.project)
    if args.name is None:
        raise ValueError("Train run name does not specified")
    trainer.train(args.name)
