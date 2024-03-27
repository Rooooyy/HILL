#!/usr/bin/env python
# coding:utf-8
import torch
import random
import numpy as np

import json
import os

from transformers import BertConfig


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def name_join(*args: str, tab='_'):
    name = ""
    for key in args:
        name += str(key)
        name += tab
    return name.rstrip(tab)


class Configure(object):
    def __init__(self, config=None, config_json_file=None):
        """
        convert conf.json to Dict and Object
        :param config: Dict, change specified configure
        :param config_json_file: conf.json, json.load(f)
        """
        if config_json_file:
            assert os.path.isfile(config_json_file), "Error: Configure file not exists!!"
            with open(config_json_file, 'r') as fin:
                self.dict = json.load(fin)
            self.update(self.dict)
        if config:
            self.update(config)

    def __getitem__(self, key):
        """
        get configure as attribute
        :param key: specified key
        :return: configure value -> Int/List/Dict
        """
        return self.__dict__[key]

    def __contains__(self, key):
        """
        check whether the configure is set
        :param key: specified key
        :return: Boolean
        """
        return key in self.dict.keys()

    def add(self, k, v):
        """
        add new configure
        :param k: specified key
        :param v: value
        """
        self.__dict__[k] = v

    def items(self):
        """
        :return: Iteration[Tuple(Str(key), value)]
        """
        return self.dict.items()

    def update(self, config):
        """
        update configure
        :param config: Dict{k:v}
        """
        assert isinstance(config, dict), "Configure file should be a json file and be transformed into a Dictionary!"
        for k, v in config.items():
            if isinstance(v, dict):
                config[k] = Configure(v)
            elif isinstance(v, list):
                config[k] = [Configure(x) if isinstance(x, dict) else x for x in v]
        self.__dict__.update(config)