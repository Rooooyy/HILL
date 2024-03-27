import time

from transformers import AutoTokenizer
'''
Suggested version==0.10.2  
When using Windows, please make sure you have installed MSVC Redistributable and SDK before installing fairseq from source.
'''
from fairseq.data import data_utils
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model.optim import ScheduledOptim, Adam
import os
from eval import evaluate
from model.contrast import ContrastModel, StructureContrast, GraphContrast

import utils
import arg_parser

import warnings
warnings.filterwarnings('ignore')


class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None):
        self.device = device
        super(BertDataset, self).__init__()
        self.data = data_utils.load_indexed_dataset(
            data_path + '/tok', None, 'mmap'
        )
        self.labels = data_utils.load_indexed_dataset(
            data_path + '/Y', None, 'mmap'
        )
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        data = self.data[item][:self.max_token].to(self.device)
        labels = self.labels[item].to(self.device)
        return {'data': data, 'label': labels, 'idx': item, }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # new
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        max_token = max([len(b['data']) for b in batch])
        max_token = min([max_token, self.max_token])
        data = torch.full([len(batch), max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        if not os.path.exists(os.path.join(args.ckpt_dir, args.name)):
            os.makedirs(os.path.join(args.ckpt_dir, args.name))

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret


def params_and_time(config, train_set, test_set, num_epochs=20):
    """
    Experiments for parameters and training time (Figure 4)
    """
    from thop import clever_format
    config.update(vars(args))

    model = MODELS[config.model_name].from_pretrained(bert_file, num_labels=num_class, local_config=config)
    model.to(device)

    train_set = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    test_set = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    num_params = sum(p.numel() for n, p in model.named_parameters() if 'bert' not in n)
    num_params = clever_format([num_params], "%.4f")
    print("Trainable parameters: {}.".format(num_params))

    optimizer = Adam(model.parameters(), lr=config.train.bert_lr)

    begin_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for data, label, idx in train_set:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, labels=label, return_dict=True, )
            optimizer.zero_grad()
            output['loss'].backward()
            optimizer.step()
    train_time = (time.time() - begin_time) / float(num_epochs)
    print("Average train time: {} secs.".format(train_time))

    begin_time = time.time()
    model.eval()
    with torch.no_grad():
        for data, label, idx in test_set:
            padding_mask = data != tokenizer.pad_token_id
            _ = model(data, padding_mask, return_dict=True,)
    inf_time = time.time() - begin_time
    print("Inference time: {} secs".format(inf_time))


def run_test(config, test_set, extra):
    checkpoint = torch.load(os.path.join(config.ckpt_dir, config.name, 'best_{}.pt'.format(extra)),
                            map_location='cpu')
    model = MODELS[config.model_name].from_pretrained(bert_file, num_labels=num_class, local_config=config)
    model.load_state_dict(checkpoint['param'])
    model.to(device)

    test_set = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    truth = []
    pred = []

    model.eval()
    with torch.no_grad():
        for data, label, idx in test_set:
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

    scores = evaluate(pred, truth, label_dict)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('Test performance with best_val%s ↓\nmicro-f1: %.4f\nmacro-f1: %.4f' % (extra, micro_f1, macro_f1))


def run_train(config, train_set, dev_set):
    model = MODELS[config.model_name].from_pretrained(bert_file, num_labels=num_class, local_config=config)
    if config.wandb:
        import wandb
        wandb.init(config=config, project='htc')
        wandb.watch(model)

    train_set = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    dev_set = DataLoader(dev_set, batch_size=config.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # Dual optimizer
    bert_param = [p for n, p in model.named_parameters() if 'bert' in n]
    hill_param = [p for n, p in model.named_parameters() if 'bert' not in n]
    if config.train.warmup:
        bert_optimizer = ScheduledOptim(Adam(bert_param, lr=config.train.bert_lr),
                                        config.train.bert_lr,
                                        n_warmup_steps=config.train.warmup_steps)
    else:
        bert_optimizer = Adam(bert_param, lr=config.train.bert_lr)

    hill_optimizer = Adam(hill_param, lr=config.learning_rate)

    model.to(device)
    save = Saver(model, bert_optimizer, None, config)

    best_score_macro = 0.0
    best_score_micro = 0.0
    early_stop_count = 0

    for epoch in range(config.train.epoch):
        if early_stop_count >= config.train.early_stop:
            print("Early stop!")
            break
        model.train()
        loss = 0.0
        # -------------Train-----------------------------
        for data, label, idx in train_set:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, labels=label, return_dict=True, )
            loss += output['loss'].item()

            bert_optimizer.zero_grad()
            hill_optimizer.zero_grad()

            output['loss'].backward()

            bert_optimizer.step()
            hill_optimizer.step()
            torch.cuda.empty_cache()  # For nyt batch=24
        loss = loss / len(train_set) * config.batch_size
        if args.wandb:
            wandb.log({'train_loss': loss})

        # -------------Eval-------------------------------
        model.eval()
        with torch.no_grad():
            truth = []
            pred = []
            for data, label, idx in dev_set:
                padding_mask = data != tokenizer.pad_token_id
                output = model(data, padding_mask, labels=label, return_dict=True, )
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())

        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print("epoch: %d\t loss: %.6f\t micro_f1: %.4f\t macro_f1: %.4f" % (epoch, loss, micro_f1, macro_f1))
        if config.wandb:
            wandb.log({'val_micro': micro_f1, 'val_macro': macro_f1, 'best_micro': best_score_micro,
                       'best_macro': best_score_macro})
        early_stop_count += 1

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join(config.ckpt_dir, config.name, 'best_micro.pt'))
            early_stop_count = 0

        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join(config.ckpt_dir, config.name, 'best_macro.pt'))
            early_stop_count = 0


def run_once(config, train, dev, test):
    config.update(vars(args))
    # Train & Eval
    run_train(config, train, dev)
    # Test
    run_test(config, test, 'micro')
    run_test(config, test, 'macro')


if __name__ == '__main__':
    args = arg_parser.get_args()
    print(args)
    config = utils.Configure(config_json_file=os.path.join(args.cfg_dir, args.model_name + '.json'))

    bert_file = "/YOUR_BERT_DIR/bert-base-uncased"  # For offline.
    # bert_file = 'bert-base-uncased'  # For online.
    tokenizer = AutoTokenizer.from_pretrained(bert_file)
    data_path = os.path.join(args.data_dir, args.dataset)
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    MODELS = {
        'hill': StructureContrast,  # HILL (ours)
        'hgclr': ContrastModel,  # HGCLR (Wang et al.@ACL'22)
        'gclr': GraphContrast,  # Ablation model
    }
    device = config.device_setting.device

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    split = torch.load(os.path.join(data_path, 'split.pt'))

    train = Subset(dataset, split['train'])
    dev = Subset(dataset, split['val'])
    test = Subset(dataset, split['test'])

    # params_and_time(config, train, test)

    # ------------Run once----------------------
    utils.seed_torch(args.seed)
    if len(args.name) == 0:
        args.name = utils.name_join(args.begin_time,
                                    args.model_name,
                                    args.dataset,
                                    args.batch_size,
                                    args.learning_rate,
                                    args.tree_depth,
                                    args.hidden_dim,
                                    args.hidden_dropout,
                                    args.tree_pooling_type,
                                    args.lamda)

    run_once(config, train, dev, test)

