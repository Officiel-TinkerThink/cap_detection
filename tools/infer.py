from ultralytics import YOLO

def infer():
  model = YOLO("runs/detect/train/weights/best.pt")
  result = model.predict('data/images/val', save=True, name='predictions', conf=0.01)
  print(type(result))

def run_inference():
  pass

if __name__ == '__main__':
  infer()