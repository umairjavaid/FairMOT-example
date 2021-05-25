from fairmot import get_pose_net
from load_fairmot_weights import load_fairmot_weights

model = get_pose_net(34, heads={'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}, head_conv=256)
weights_path = "/content/FairMOT/models/crowdhuman_dla34.pth"
model = load_fairmot_weights(model, weights_path)
