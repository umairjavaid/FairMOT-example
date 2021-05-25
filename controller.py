from fairmot import get_pose_net

model = get_pose_net(34, heads={'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}, head_conv=256)
print(model)
