from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2 

def convert_to_PIL(img):
  img = Image.fromarray(img)
  return img

data_transforms = transforms.Compose([
  transforms.Resize(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406],[0.299, 0.244, 0.225])
  ])

def makeImgModelReady(img):
  global data_transforms
  img = convert_to_PIL(img)
  img = data_transforms(img)
  img = torch.unsqueeze(img, 0)
  return img

def convert_tensor_to_img(tensor):
    return img

def save_img(img):
    return img
    
def fairmot_transform(img_path):
  # Read image
  img0 = cv2.imread(img_path)  # BGR
  assert img0 is not None, 'Failed to load ' + img_path

  # Padded resize
  img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

  # Normalize RGB
  img = img[:, :, ::-1].transpose(2, 0, 1)
  img = np.ascontiguousarray(img, dtype=np.float32)
  img /= 255.0
  
  if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

def letterbox(img, height=608, width=1088,
  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
  shape = img.shape[:2]  # shape = [height, width]
  ratio = min(float(height) / shape[0], float(width) / shape[1])
  new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
  dw = (width - new_shape[0]) / 2  # width padding
  dh = (height - new_shape[1]) / 2  # height padding
  top, bottom = round(dh - 0.1), round(dh + 0.1)
  left, right = round(dw - 0.1), round(dw + 0.1)
  img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
  return img, ratio, dw, dh
