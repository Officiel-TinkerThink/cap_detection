# models/yolo_model.py
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import intersect_dicts


def get_detection_model(
    num_classes: int = 3, checkpoint_path: str = None, pretrained: bool = True
):
    """
    Create YOLO model with optional checkpoint loading.

    Args:
        num_classes: Number of classes
        checkpoint_path: Path to checkpoint from previous training
        pretrained: Load COCO pretrained backbone if no checkpoint

    Returns:
        model: Initialized model
        checkpoint: Dict containing training state (if checkpoint loaded)
    """
    model = DetectionModel(cfg="yolov8n.yaml", nc=num_classes)
    checkpoint = None

    if checkpoint_path and checkpoint_path.exists():
        # Scenario 1: Resume/fine-tune from YOUR checkpoint
        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Load full model state (backbone + head)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print("✅ Loaded full model from checkpoint")

    elif pretrained:
        # Scenario 2: Load COCO pretrained backbone only
        try:
            ckpt = torch.load("yolov8n.pt", map_location="cpu", weights_only=False)
            weights = intersect_dicts(
                ckpt["model"].state_dict(), model.state_dict(), exclude=["head"]
            )
            model.load_state_dict(weights, strict=False)
            print("✅ Loaded COCO pretrained backbone")
        except FileNotFoundError:
            print("⚠️ yolov8n.pt not found, training from scratch")

    return model
