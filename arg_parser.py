import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-d', '--dataset', type=str, default='wos', choices=['wos', 'nyt', 'rcv1'],
                        help='Dataset.')
    # model
    parser.add_argument('-mn', '--model_name', type=str, default="hill")

    # training
    parser.add_argument('-n', '--name', type=str, default='', help='Name for a specific run.')
    parser.add_argument('-s', '--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16).')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4).')
    parser.add_argument('-l2', '--l2_rate', type=float, default=0.01,
                        help='L2 penalty (default: 0.01)')  # When using AdamW, 0.01 always works.
    parser.add_argument('-l', '--lamda', default=0.05, type=float,
                        help='The weight of contrastive loss (default: 0.05).')
    parser.add_argument('-f', '--freeze', default=False, action='store_true', help='Freeze BERT or not.')
    parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')
    # HRLEncoder
    parser.add_argument('-k', '--tree_depth', type=int, default=2,
                        help='The depth of the coding tree (default: 2).')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=512,
                        help='The number of hidden units in HRLConv (default: 512).')
    parser.add_argument('-dp', '--hidden_dropout', type=float, default=0.5,
                        help='Dropout ratio for readout layer (default: 0.5)')
    parser.add_argument('-tp', '--tree_pooling_type', type=str, default="sum", choices=["root", "sum", "avg", "max"],
                        help='Pooling for over nodes in a tree: root, sum, or average')
    parser.add_argument('-ho', '--hrl_output', type=str, default='bert', choices=['bert', 'tree', 'residual', 'concat'])
    # GNNEncoder (for ablation)
    parser.add_argument('-gc', '--graph_conv', type=str, default='GIN', choices=['GCN', 'GAT', 'GIN'])
    parser.add_argument('-gp', '--graph_pooling_type', type=str, default='sum', choices=['sum', 'avg', 'max'])
    parser.add_argument('-gl', '--conv_layers', type=int, default=3)
    parser.add_argument('-go', '--graph_output', type=str, default='graph', choices=['graph', 'tree', 'concat'])
    parser.add_argument('--residual', default=False, action='store_true')

    # contrast
    # moved to config/xxx.json
    # parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
    # hgclr
    # parser.add_argument('--thre', default=0.02, type=float,
    #                     help='Threshold for keeping tokens. Denote as gamma in the paper.') # moved to config/xxx.json
    parser.add_argument('--graph', default=True, action='store_false', help='Whether use graph encoder.')
    # parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.') # moved to config/xxx.json

    # ablation
    parser.add_argument('--contrast_loss', default=True, action='store_false', help='Whether use contrastive loss.')
    parser.add_argument('--cls_loss', default=True, action='store_false')
    parser.add_argument('--multi_label', default=True, action='store_false',
                        help='Whether the task is multi-label classification.')

    # dirs
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    parser.add_argument('--cfg_dir', type=str, default='config')

    parser.add_argument('--begin_time', type=str, default=time.strftime("%m%d_%H%M", time.localtime()))

    args = parser.parse_args()
    return args
