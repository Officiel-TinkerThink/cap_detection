from model.detection_model import get_detection_model
import tqdm
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss
from pathlib import Path
import yaml
import torch

def train():
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset_params']
    hyper_params = config['hyperparams']
    paths = config['paths']
    device = torch.device(config.get('device', 'cpu'))
    
    # Create save directory
    save_dir = Path(paths['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    if checkpoint_path:
        model = get_detection_model(num_classes=3, checkpoint_path=checkpoint_path)
    else:
        model = get_detection_model(num_classes=3, pretrained=True)
    model = model.to(device)
    model.train()
    
    # Dataset
    data_info = check_det_dataset(paths['data_config'])
    train_dataset = YOLODataset(
        img_path=data_info['train'],
        imgsz=hp['imgsz'],
        batch_size=hp['batch_size'],
        augment=True,
        hyp=hp
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    # Loss & optimizer
    criterion = v8DetectionLoss(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hp['lr0'],
        momentum=hp['momentum'],
        weight_decay=hp['weight_decay']
    )
    
    # Training loop
    for epoch in range(hp['epochs']):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hp['epochs']}")
        
        for batch in pbar:
            images = batch['img'].to(device, non_blocking=True).float() / 255.0
            targets = batch['cls'].to(device), batch['bboxes'].to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss, loss_items = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'box': f"{loss_items[0].item():.4f}",
                'cls': f"{loss_items[1].item():.4f}"
            })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch+1}.pt")
    
    # Final save
    torch.save(model.state_dict(), save_dir / "final.pt")
    print(f"\n✅ Training complete! Model saved to {save_dir}")

if __name__ == '__main__':
    train()