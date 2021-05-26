from fairmot import get_pose_net
from load_fairmot_weights import load_fairmot_weights
from img_preprocess import *
import matplotlib.pyplot as plt

model = get_pose_net(34, heads={'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}, head_conv=256)
weights_path = "/content/FairMOT/models/crowdhuman_dla34.pth"
model = load_fairmot_weights(model, weights_path)
img_path = "/content/FairMOT-example/cover4.jpg"

img = plt.imread(img_path)
print(img.shape)

img = makeImgModelReady(img)

print(model(img))
