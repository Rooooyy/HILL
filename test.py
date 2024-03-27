from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
import argparse
import os
from train import BertDataset
from eval import evaluate
from model.contrast import ContrastModel, StructureContrast

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size.')
parser.add_argument('-n', '--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('-e', '--extra', default='_macro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join('ckpt', args.name, 'best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    device = args.device
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    data_path = os.path.join(args.data_dir, args.dataset)

    if not hasattr(args, 'graph'):
        args.graph = False
    # print(args)

    config = utils.Configure(config_json_file=os.path.join(args.cfg_dir, args.model_name + '.json'))
    config.update(vars(args))
    bert_file = "/YOUR_BERT_DIR/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(bert_file)

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    MODELS = {
        'hill': StructureContrast,
        'hgclr': ContrastModel,
    }
    model = MODELS[config.model_name].from_pretrained(bert_file, num_labels=num_class, local_config=config)

    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model.load_state_dict(checkpoint['param'])

    model.to(device)

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []

    model.eval()
    with torch.no_grad():
        for data, label, idx in test:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True, )
            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                truth.append(t)
            for l in output['logits']:
                pred.append(torch.sigmoid(l).tolist())

    # pbar.close()
    scores = evaluate(pred, truth, label_dict)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('Test performance with best_val%s ↓\nmicro-f1: %.4f\nmacro-f1: %.4f' % (extra, micro_f1, macro_f1))
