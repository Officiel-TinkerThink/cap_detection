import torch
import numpy as np
import argparse
import random
import os
import yaml
from tqdm import tqdm
from models.yolov1 import YOLOV1
from dataset.voc import VOCDataset
from utils.visualization_utils import *
from utils.utils import *
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_dataset(args, device):
  with open(args.config_path, 'r') as file:
    try:
      config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
      print(exc)
  
  dataset_config = config['dataset_params']
  model_config = config['model_params']
  train_config = config['train_params']
  
  seed = train_config['seed']
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if device == 'cuda':
      torch.cuda.manual_seed_all(seed)

  voc = VOCDataset('test', im_dir=dataset_config['im_test_path'], ann_dir=dataset_config['ann_test_path'])
  test_dataset = DataLoader(voc, batch_size=1, shuffle=False)

  yolo_model = YOLOV1(num_classes=dataset_config['num_classes'], model_config=model_config)
  yolo_model.eval()
  yolo_model.to(device)
  yolo_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                            train_config['ckpt_name']),
                                                map_location=device))
  return yolo_model, voc, test_dataset, config

def convert_yolo_pred_x1y1x2y2(yolo_pred, S, B, C, use_sigmoid=False):
  """
  Method converts yolo predictions to
  x1y1x2y2 format
  """
  out = yolo_pred.reshape((S, S, 5 * B + C))
  if use_sigmoid:
      out[..., :5 * B] = torch.nn.functional.sigmoid(out[..., :5 * B])
  out = torch.clamp(out, min=0., max=1.)
  class_score, class_idx = torch.max(out[..., 5 * B:], dim=-1)

  # Create a grid using these shifts
  # Will use these for converting x_center_offset/y_center_offset
  # values to x1/y1/x2/y2(normalized 0-1)
  # S cells = 1 => each cell adds 1/S pixels of shift
  shifts_x = torch.arange(0, S, dtype=torch.int32, device=out.device) * 1 / float(S)
  shifts_y = torch.arange(0, S, dtype=torch.int32, device=out.device) * 1 / float(S)
  shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

  boxes = []
  confidences = []
  labels = []
  for box_idx in range(B):
      # xc_offset yc_offset w h -> x1 y1 x2 y2
      boxes_x1 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) -
                  0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
      boxes_y1 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) -
                  0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
      boxes_x2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) +
                  0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
      boxes_y2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_y) +
                  0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
      boxes.append(torch.cat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1))
      confidences.append((out[..., 4 + box_idx * 5] * class_score).reshape(-1))
      labels.append(class_idx.reshape(-1))
  boxes = torch.cat(boxes, dim=0)
  scores = torch.cat(confidences, dim=0)
  labels = torch.cat(labels, dim=0)
  return boxes, scores, labels

def infer(args):
  if not os.path.exists('samples'):
    os.mkdir('samples')
  yolo_model, voc, test_dataset, config = load_model_and_dataset(args, device=device)
  conf_threshold = config['train_params']['infer_conf_threshold']
  nms_threshold = config['train_params']['nms_threshold']

  num_samples = 5

  for i in tqdm(range(num_samples)):
    dataset_idx = random.randint(0, len(voc))
    im_tensor, targets, fname = voc[dataset_idx]

    out = yolo_model(im_tensor.unsqueeze(0).to(device))
    boxes, scores, labels = convert_yolo_pred_x1y1x2y2(out,
                    S=yolo_model.S,
                    B=yolo_model.B,
                    C=yolo_model.C,
                    use_sigmoid=config['model_params']['use_sigmoid'])
    
    # Confidence Score Thresholding
    keep = torch.where(scores > conf_threshold)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # NMS
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(labels):
      curr_indices = torch.where(labels == class_id)[0]
      curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
                                                    scores[curr_indices],
                                                    nms_threshold)
      keep_mask[curr_indices[curr_keep_indices]] = True
    keep = torch.where(keep_mask)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Visualization

    if not os.path.exists('samples/pred'):
      os.mkdir('samples/preds')
    if not os.path.exists('samples/grid_cls'):
      os.mkdir('samples/grid_cls')
    
    im = cv2.imread(fname)
    h, w = im.shape[:2]
    # Scale prediction boxes x1y1x2y2 from 0-1 to 0-w and 0-h
    boxes[..., 0::2] = (w * boxes[..., 0::2])
    boxes[..., 1::2] = (w * boxes[..., 1::2])

    out_img = visualize(image=im, bboxes=boxes.detach().cpu().numpy(),
                        category_ids=labels.detach().cpu().numpy(),
                        category_id_to_name=voc.idx2label,
                        scores=scores.detach().cpu().numpy())
    
    cv2.imwrite('samples/preds/{}_pred.jpg'.format(i), out_img)

    # Below lines of code are only for drawing class prob map
    im = cv2.resize(im, (yolo_model.im_size, yolo_model.im_size))

    # Draw a SxS grid on image
    grid_im = draw_grid(im, (yolo_model.S, yolo_model.S))

    out = out.reshape((yolo_model.S, yolo_model.S, 5 * yolo_model.B + yolo_model.C))
    cls_val, cls_idx = torch.max(out[..., 5 * yolo_model.B:], dim=-1)

    # Draw colored squares for probability mappings on image
    rect_im = draw_cls_grid(im, cls_idx, (yolo_model.S, yolo_model.S))
    # Draw grid again on top of this image
    rect_im = draw_grid(rect_im, (yolo_model.S, yolo_model.S))

    # Overlay image with grid and cls mapping with grid on top of each other
    res = cv2.addWeighted(rect_im, 0.5, grid_im, 0.5, 1.0)
    # Write Class labels on grid on this image
    res = draw_cls_text(res, cls_idx, voc.idx2label, (yolo_model.S, yolo_model.S))
    cv2.imwrite('samples/grid_cls/{}_grid_map.jpg'.format(i), res)
  print('Done Detecting...')
    

def evaluate_map(args):
  yolo_model, voc, test_dataset, config = load_model_and_dataset(args, device)
  conf_threshold = config['train_params']['infer_conf_threshold']
  nms_threshold = config['train_params']['nms_threshold']
  
  gts = []
  preds = []
  difficults = []
  for im_tensor, target, fname in tqdm(test_dataset):
    im_tensor = im_tensor.float().to(device)
    target_boxes = target['bboxes'].float().to(device)[0]
    target_labels = target['labels'].long().to(device)[0]
    difficult = target['difficult'].long().to(device)[0]
    out = yolo_model(im_tensor)
    boxes, scores, labels = convert_yolo_pred_x1y1x2y2(out,
                    S=yolo_model.S,
                    B=yolo_model.B,
                    C=yolo_model.C,
                    use_sigmoid=config['model_params']['use_sigmoid'])

    # Confidence Score Thresholding
    keep = torch.where(scores > conf_threshold)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # NMS
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(labels):
      curr_indices = torch.where(labels == class_id)[0]
      curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
                                                    scores[curr_indices],
                                                    nms_threshold)
      keep_mask[curr_indices[curr_keep_indices]] = True
    keep = torch.where(keep_mask)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    pred_boxes = {}
    gt_boxes = {}
    difficult_boxes = {}
    for label_name in voc.label2idx:
      pred_boxes[label_name] = []
      gt_boxes[label_name] = []
      difficult_boxes[label_name] = []

    for idx, box in enumerate(boxes):
      x1, y1, x2, y2 = box.detach().cpu().numpy()
      label = labels[idx].detach().cpu().item()
      score = scores[idx].detach().cpu().item()
      label_name = voc.idx2label[label]
      pred_boxes[label_name].append([x1, y1, x2, y2, score])
    for idx, box in enumerate(target_boxes):
      x1, y1, x2, y2 = box.detach().cpu().numpy()
      label = target_labels[idx].detach().cpu().item()
      label_name = voc.idx2label[label]
      gt_boxes[label_name].append([x1, y1, x2, y2])
      difficult_boxes[label_name].append(difficult[idx].detach().cpu().item())
    
    gts.append(gt_boxes)
    preds.append(pred_boxes)
    difficults.append(difficult_boxes)

  mean_ap, all_aps = compute_map(gts, preds, method='interp', difficult=difficults)
  print('Class Wise Average Precisions')
  for idx in range(len(voc.idx2label)):
    print('AP for class {} : {:.4f}'.format(voc.idx2label[idx], all_aps[voc.idx2label[idx]]))
  print('Mean Average Precision : {:.4f}'.format(mean_ap))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Arguments forFaster yolov1 Inference')
  parser.add_argument('--config', dest='config_path', 
                      default='config/voc.yaml', type=str)
  parser.add_argument('--evaluate', dest='evaluate', default=False, type=bool)
  parser.add_argument('--infer_samples', dest='infer_samples',
                      default=True, type=bool)
  parser.add_argument('--num_samples', dest='num_samples',
                      default=1, type=int)
  args = parser.parse_args()

  with torch.no_grad():
    if args.infer_samples:
      infer(args)
    else:
      print('Not Inferring for sample as `infer_samples` argument is set to False')
    if args.evaluate:
      evaluate_map(args)
    else:
      print('Not Evaluating as `evaluate` argument is set to False')