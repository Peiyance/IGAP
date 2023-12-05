import torch
from models.sage import GraphSage

input_dim = 2879
hidden_dim = 128
gnn_layer_num = 2
dataset_name = "Cora_ML"
gnn_type = "SAGE"
epoch=200
load_ckpt_path = "./pt_ckpt/{}/Graph_CL_{}_{}.pth".format(dataset_name, gnn_type, str(epoch))

model = GraphSage(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, layer_num=gnn_layer_num, pool=None)
print(load_ckpt_path)
model.load_state_dict(torch.load(load_ckpt_path))
print("Load Done")