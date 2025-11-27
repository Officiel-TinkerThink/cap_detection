import typer
from ultralytics import YOLO
from pathlib import Path
from typing_extensions import Optional

app = typer.Typer()


@app.command()
def train(config: Path = 'config/settings.yaml') -> None:
    """
    Execute model training using the specified configuration file.
    
    This command initializes a Trainer instance and starts the training pipeline
    with parameters defined in the configuration file.
    
    Args:
        config: Path to the YAML configuration file containing training parameters.
    """
    from tools.train import Trainer

    trainer = Trainer(config)
    trainer.train()

@app.command()
def infer(
  config: Path = 'config/settings.yaml',
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
    if image is None:
      raise ValueError('Image Argument is not specified')
    model = YOLO("runs/detect/train/weights/best.pt")
    result = model.predict("data/images/val", save=True, name="inference")


def main() -> None:
    """
    Entry point for the CLI application.
    
    This function executes the typer CLI application with its registered commands.
    """
    app()


if __name__ == "__main__":
    main()