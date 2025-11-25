# dataset/yolo_dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLODataset(Dataset):
    """
    YOLO format dataset for object detection.
    Expects: images/ and labels/ directories with same filename structure.
    Label format: class_id cx cy w h (normalized 0-1)
    """
    
    def __init__(self, img_path, imgsz=640, augment=True, hyp=None):
        """
        Args:
            img_path: Path to images directory (e.g., './dataset/images/train')
            imgsz: Target image size for resizing
            augment: Whether to apply augmentations
            hyp: Hyperparameters dict for augmentation settings
        """
        self.img_path = Path(img_path)
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp or {}
        
        # Find all images
        self.img_files = sorted(list(self.img_path.glob('*.jpg')))
        
        if not self.img_files:
            raise FileNotFoundError(f"No images found in {img_path}")
        
        # Build label paths
        self.label_files = [
            Path(f).with_name(f.name.replace('.jpg', '.txt'))
            for f in self.img_files
        ]

        self.label2idx = {'light blue' : 0, 'dark blue': 1, 'others': 2}
        self.idx2label = {value:key for key, value in self.label2idx.items()}

        # Albumentations transforms
        if self.augment:
            self.transform = A.Compose([
                A.RandomScale(scale_limit=0.5, p=0.5),
                A.PadIfNeeded(min_height=imgsz, min_width=imgsz, 
                             border_mode=cv2.BORDER_CONSTANT, value=114),
                A.RandomCrop(height=imgsz, width=imgsz, p=0.5),
                A.HorizontalFlip(p=self.hyp.get('flip_p', 0.5)),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.Blur(blur_limit=3, p=0.1),
                A.Resize(height=imgsz, width=imgsz, p=1),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_area=0,
                min_visibility=0
            ))
        else:
            # Validation transform (only resize & pad)
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=imgsz, min_width=imgsz, 
                             border_mode=cv2.BORDER_CONSTANT, value=114),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_area=0,
                min_visibility=0
            ))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Returns {'img': tensor, 'cls': tensor, 'bboxes': tensor, 'batch_idx': idx}"""
        
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load labels
        label_path = self.label_files[idx]
        if label_path.exists():
            # Parse YOLO format: class_id cx cy w h
            labels = np.loadtxt(label_path, delimiter=' ', ndmin=2)
            if len(labels) == 0:
                # Empty image
                bboxes = np.zeros((0, 4), dtype=np.float32)
                class_labels = np.zeros((0, 1), dtype=np.float32)
            else:
                class_labels = labels[:, 0:1]
                bboxes = labels[:, 1:5]  # cx cy w h (already normalized)
        else:
            # No labels for this image
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros((0, 1), dtype=np.float32)
        
        # Apply augmentations
        if self.augment:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels.flatten().tolist()
            )
        else:
            # For validation, just resize and pad to imgsz
            ratio = min(self.imgsz / h, self.imgsz / w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            
            img_resized = cv2.resize(img, (new_w, new_h))
            pad_h = (self.imgsz - new_h) // 2
            pad_w = (self.imgsz - new_w) // 2
            
            transformed = self.transform(
                image=img_resized,
                bboxes=bboxes,  # Keep normalized coords, transform handles scaling
                class_labels=class_labels.flatten().tolist()
            )
        
        # Extract transformed data
        img_tensor = transformed['image']  # Already [C, H, W] after ToTensorV2
        bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        class_labels = np.array(transformed['class_labels'], dtype=np.float32)
        
        # Convert to tensors
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0, 1), dtype=torch.float32)
        else:
            bboxes = torch.from_numpy(bboxes)
            class_labels = torch.from_numpy(class_labels).unsqueeze(1)
        
        return {
            'img': img_tensor,
            'cls': class_labels,  # Shape: [N, 1]
            'bboxes': bboxes,     # Shape: [N, 4] in yolo format (cx cy w h)
            'batch_idx': idx,
            'img_path': str(img_path)
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate samples into batches with variable-length targets"""
        imgs = torch.stack([item['img'] for item in batch], dim=0)
        batch_idx = []
        cls = []
        bboxes = []
        
        for i, item in enumerate(batch):
            if len(item['cls']) > 0:
                batch_idx.append(torch.full((item['cls'].shape[0], 1), i, dtype=torch.long))
                cls.append(item['cls'])
                bboxes.append(item['bboxes'])
        
        if cls:
            batch_idx = torch.cat(batch_idx, dim=0)
            cls = torch.cat(cls, dim=0)
            bboxes = torch.cat(bboxes, dim=0)
        else:
            # Empty batch (no objects)
            batch_idx = torch.zeros((0, 1), dtype=torch.long)
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            'img': imgs,          # [B, 3, H, W]
            'batch_idx': batch_idx, # [N, 1] - which image each target belongs to
            'cls': cls,           # [N, 1] - class IDs
            'bboxes': bboxes       # [N, 4] - normalized cx cy w h
        }