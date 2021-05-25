from PIL import Image
from torchvision import transforms

def convert_to_PIL(img):
  img = Image.fromarray(img)
  return img

data_transforms = transforms.compose([
  transforms.ToTensor(),
  transforms.Resize(224),
  transforms.Normalize([0.485, 0.456, 0.406],[0.299, 0.244, 0.225])

def makeImgModelReady(img):
  global data_transforms
  img = convert_to_PIL(img)
  img = data_transforms(img)
  img = torch.unsqueeze(img, 0)
  return img
