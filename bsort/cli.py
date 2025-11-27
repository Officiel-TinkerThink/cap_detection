from pathlib import Path
from typing import Optional

import typer
import yaml
from ultralytics import YOLO

from tools.train import Trainer

app = typer.Typer()


@app.command()
def train(config: Path = "configs/settings.yaml") -> None:
    """
    Execute model training using the specified configuration file.

    This command initializes a Trainer instance and starts the training pipeline
    with parameters defined in the configuration file.

    Args:
        config: Path to the YAML configuration file containing training parameters.
    """

    trainer = Trainer(config)
    trainer.train()


@app.command()
def infer(
    config: Path = "configs/settings.yaml",
    image: Optional[Path] = None,
) -> None:
    """
    Run inference using a trained model on specified image(s).

    This command loads a trained YOLO model and runs predictions on either a
    single image or a directory of images.

    Args:
        config: Path to the YAML configuration file containing inference settings.
        image: path to image (file or directory) for inference.
    """
    with open(config, "r", encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)
        infer_params: dict = config.get("infer")

    model_path = infer_params.pop("model_path")

    if image is None:
        raise ValueError("Image Argument is not specified")
    model = YOLO(model_path)
    model.predict(image, save=True, **infer_params)


def main() -> None:
    """
    Entry point for the CLI application.

    This function executes the typer CLI application with its registered commands.
    """
    app()


if __name__ == "__main__":
    main()
