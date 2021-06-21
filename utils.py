import numpy as np
import json
import math
from concurrent import futures
from torch.utils.data import Dataset
import time

def parse_json(args):
    lines, trunc = args
    return [_[:trunc] for _ in [json.loads(_) for _ in lines[0]]], \
           [_[:trunc] for _ in [json.loads(_) for _ in lines[1]]]


def collate_fn(batch):
    claims = [_[0] for _ in batch]
    bios = [_[1] for _ in batch]

    key = np.array([len(_) for _ in claims]).argsort()

    claims = [claims[_] for _ in reversed(key)]
    bios = [bios[_] for _ in reversed(key)]
    length = [len(_) for _ in claims]

    ml = len(claims[0])
    claims = [_ + [0 for _ in range(ml - len(_))] for _ in claims]
    bios = [_ + [0 for _ in range(ml - len(_))] for _ in bios]
    return claims, bios, length


class dataset(Dataset):
    def __init__(self, mode='train', worker=1, trunc=999999999999):
        if mode not in ['train', 'test', 'val']:
            print('dataset mode should be in [train, test, val]')
            exit()

        with open('./data/%s_emjeol_input_cased.jsonl' % mode, encoding='utf-8') as f, \
                open('./data/%s_bio.jsonl' % mode, encoding='utf-8') as f2:
            self.claims = []
            self.bios = []
            if worker > 1:
                with futures.ProcessPoolExecutor(worker) as pool:
                    claims = f.readlines()
                    bios = f2.readlines()
                    n = len(claims)
                    for _ in pool.map(parse_json,
                                      [([claims[math.ceil(n / worker) * i: math.ceil(n / worker) * (i + 1)],
                                        bios[math.ceil(n / worker) * i: math.ceil(n / worker) * (i + 1)]], trunc)
                                       for i in range(worker)]):
                        c, b = _
                        self.claims += c
                        self.bios += b
            else:
                for c, b in zip(f, f2):
                    c = json.loads(c)
                    b = json.loads(b)
                    self.claims.append(c)
                    self.bios.append(b)

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        return self.claims[idx], self.bios[idx]


class dataset_je(Dataset):
    def __init__(self, mode='train', worker=1, trunc=999999999999):
        if mode not in ['train', 'test', 'val']:
            print('dataset mode should be in [train, test, val]')
            exit()

        with open('./data/%s_emjeol_input_cased.jsonl' % mode, encoding='utf-8') as f, \
                open('./data/%s_bio_just_entity_or_not.jsonl' % mode, encoding='utf-8') as f2:
            self.claims = []
            self.bios = []
            if worker > 1:
                with futures.ProcessPoolExecutor(worker) as pool:
                    claims = f.readlines()
                    bios = f2.readlines()
                    n = len(claims)
                    for _ in pool.map(parse_json,
                                      [([claims[math.ceil(n / worker) * i: math.ceil(n / worker) * (i + 1)],
                                         bios[math.ceil(n / worker) * i: math.ceil(n / worker) * (i + 1)]], trunc)
                                       for i in range(worker)]):
                        c, b = _
                        self.claims += c
                        self.bios += b
            else:
                for c, b in zip(f, f2):
                    c = json.loads(c)
                    b = json.loads(b)
                    self.claims.append(c)
                    self.bios.append(b)

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        return self.claims[idx], self.bios[idx]

if __name__ == '__main__':
    a = dataset('train', 20)

    print()