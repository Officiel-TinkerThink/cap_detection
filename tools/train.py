from model.detection_model import get_detection_model
from tqdm import tqdm
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss
from dataset.dataset import YOLODataset
from pathlib import Path
import argparse
import yaml
import torch
from types import SimpleNamespace

def train(args):
    # Load config
    config_path = args.config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    hp = config['hyperparams']
    data_config = config['dataset_params']
    device = torch.device(config.get('device', 'cpu'))
    
    # Create save directory
    save_dir = Path(hp['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        model = get_detection_model(num_classes=3, checkpoint_path=args.checkpoint_path)
    else:
        model = get_detection_model(num_classes=3, pretrained=True)
    model = model.to(device)
    model.train()
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5,
        lr0=hp['lr'], lrf=0.01, imgsz=data_config['imgsz'],
        augment=True
    )
    
    # Dataset
    train_dataset = YOLODataset(
        img_path=data_config['img_path'],
        imgsz=data_config['imgsz'],
        augment=True,
        hyp=data_config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp['batch_size'],
        shuffle=True,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    # Loss & optimizer
    criterion = v8DetectionLoss(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp['lr'],
        weight_decay=hp['weight_decay']
    )

    
    # Training loop
    for epoch in range(hp['num_epochs']):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hp['num_epochs']}")
        
        for batch in pbar:
            # ✅ Move ENTIRE batch dict to device
            batch_device = {
                'img': batch['img'].to(device, non_blocking=True).float() / 255.0,
                'batch_idx': batch['batch_idx'].to(device, non_blocking=True),
                'cls': batch['cls'].to(device, non_blocking=True),
                'bboxes': batch['bboxes'].to(device, non_blocking=True)
            }
            
            # Forward
            optimizer.zero_grad()
            preds = model(batch_device['img'])
            loss_tensor, loss_components = criterion(preds, batch_device)

            # ✅ Sum across loss types to get scalar
            total_loss = loss_tensor.sum()  # box + cls + dfl

            # Backward & step
            total_loss.backward()
            optimizer.step()

            # ✅ Log individual components (divide by batch_size for per-sample values)
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'box': f"{loss_components[0].item():.4f}",
                'cls': f"{loss_components[1].item():.4f}",
                'dfl': f"{loss_components[2].item():.4f}"
            })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_dir / hp['save_path'])
    
    # Final save
    torch.save(model.state_dict(), save_dir / "final.pt")
    print(f"\n✅ Training complete! Model saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint file')
    args = parser.parse_args()
    train(args)