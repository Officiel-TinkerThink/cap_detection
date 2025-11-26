import argparse
import os

import yaml
from typing_extensions import Any, Dict
from ultralytics import YOLO

import wandb


class Trainer:
    def __init__(self, config_path, wandb_project):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config: Dict[str, Any] = yaml.safe_load(f)
            # train config are contain of dynamic hyperparams namely epochs, batch, imgsz, others
            self.train_config: Dict[str, Any] = config.get("train")
            # aug config are contain of static augmentations params namely, hsv_h, hsv_s, hsv_v, others
            self.aug_config: Dict[str, Any] = {}  # config.get('aug')
        else:
            raise ValueError("Config file does not exist")
        self.wandb_project = wandb_project

    def train(self, run_name):
        # inialize wandb run
        wandb.init(
            project=self.wandb_project,
            config=self.train_config,
            name=run_name,
            mode="online",
        )

        model = YOLO(self.train_config.get("model", "yolo8n.pt"))
        model.train(**self.train_config, **self.aug_config)

        # Export the model into 'ncnn'
        target = self.export(model, "ncnn")

        # export file as the output of wandb
        wandb.log({"Model Exported Target": target})
        wandb.finish()

    def export(self, model, format="ncnn", **kwargs):
        try:
            target = model.export(format=format, **kwargs)
            print(f"Model Succesfully Exported")
        except ValueError as e:
            raise ValueError("Export failed")
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
        raise Error("Train run name does not specified")
    trainer.train(args.name)
