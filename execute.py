import torch
import torch.nn as nn

from utils.pt_data_load import pt_data_loader
from utils.ft_data_load import ft_data_loader
from utils.get_args import get_args

from pretrain.graph_cl import GraphCL
from finetune.graph_prompt_augmentation import GraphPromptAugment
from finetune.graph_finetune import GraphFineTune
from finetune.inductive_graph_prompt import InductiveGraphFineTune
from finetune.graph_scratch import GraphScratch

def finetune(args):
    print("fine-tuning dataset name is {}".format(args.ft_dataset_name))
    ft_graph, input_dim, ft_train_nids, ft_test_nids, ft_eval_nids, num_class = ft_data_loader(args.ft_dataset_name)
    
    data_config = dict()
    train_config = dict()

    print(ft_graph, input_dim, num_class)
    ###### pt dataset settings
    data_config["graph"] = ft_graph
    data_config["ft_train_nids"] = ft_train_nids
    data_config["ft_test_nids"] = ft_test_nids
    data_config["ft_eval_nids"] = ft_eval_nids
    data_config["ft_batch_size"] = args.ft_batch_size
    data_config["dataset_name"] = args.ft_dataset_name
    data_config["gnn_type"] = args.gnn_type
    data_config["pt_dataset_name"] = args.pt_dataset_name
    data_config["load_pt_ckpt"] = args.load_pt_ckpt

    if args.ft_type is "graph_prompt_augmentation":
        data_config["ft_aug_type_list"] = args.ft_aug_type_list
        data_config["ft_aug_ratio_list"] = args.ft_aug_ratio_list
        data_config["ft_aug_num"] = args.ft_aug_num

    ###### train settings
    train_config["epochs"] = args.ft_epochs
    train_config["lr"] = args.ft_lr
    train_config["weight_decay"] = args.ft_weight_decay
    train_config["ckpt_epoch"] = args.ft_ckpt_epoch
    train_config["opt_type"] = args.ft_opt_type
    train_config["ft_froze"] = args.ft_froze
    train_config["device"] = args.device
    
    if args.ft_type is "graph_prompt_augmentation":
        ft_trainer = GraphPromptAugment(gnn_type=args.gnn_type, input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, gnn_layer_num=args.gnn_layer_num)
    elif args.ft_type is "inductive_graph_prompt":
        ft_trainer = InductiveGraphFineTune(gnn_type=args.gnn_type, input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, gnn_layer_num=args.gnn_layer_num, num_class=num_class)
    elif args.ft_type is "graph_scratch":
        ft_trainer = GraphScratch(gnn_type=args.gnn_type, input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, gnn_layer_num=args.gnn_layer_num, num_class=num_class)
    else:
        ft_trainer = GraphFineTune(gnn_type=args.gnn_type, input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, gnn_layer_num=args.gnn_layer_num, num_class=num_class)

    print("Fine-tuning begins!")
    ft_trainer.do_finetune(data_config, train_config)
    print("Fine-tuning finished!")

def pretrain(args):
    pt_graph_list, input_dim = pt_data_loader(args.pt_dataset_name, args.pt_graph_split_num)

    data_config = dict()
    train_config = dict()

    ###### pt dataset settings
    data_config["graph_list"] = pt_graph_list
    data_config["batch_size"] = args.pt_batch_size
    data_config["aug1"] = args.pt_aug1
    data_config["aug2"] = args.pt_aug2
    data_config["aug_ratio"] = args.pt_aug_ratio
    data_config["dataset_name"] = args.pt_dataset_name
    data_config["gnn_type"] = args.gnn_type

    ###### train settings
    train_config["epochs"] = args.pt_epochs
    train_config["lr"] = args.pt_lr
    train_config["weight_decay"] = args.pt_weight_decay
    train_config["ckpt_epoch"] = args.pt_ckpt_epoch
    train_config["opt_type"] = args.pt_opt_type
    train_config["device"] = args.device
    
    if args.pt_type is "GraphCL":
        pt_trainer = GraphCL(gnn_type=args.gnn_type, input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, gnn_layer_num=args.gnn_layer_num)
    else:
        pt_trainer = GraphCL(gnn_type=args.gnn_type, input_dim=input_dim, hidden_dim=args.gnn_hidden_dim, gnn_layer_num=args.gnn_layer_num)
    
    print("Pretrain begins!")
    pt_trainer.do_train(data_config, train_config)
    print("Pretrain Done!")


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    args = get_args()

    args.device = torch.device("cpu")
    if torch.cuda.is_available() and args.use_cuda:
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
        args.device = device
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
        args.device = device
    
    # pretrain(args)
    finetune(args)
    print("All the works have been done!")