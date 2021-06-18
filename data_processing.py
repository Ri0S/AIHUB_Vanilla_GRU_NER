import json
import os

'''
bio_just_entity_or_not.txt:
0: O
1: I
2: B

bio.txt:
0: O
1: PERSON(PS) B
2: PERSON(PS) I
3: LOCATION(LC) B
4: LOCATION(LC) I
5: ORGANIZATION(OG) B
6: ORGANIZATION(OG) I
7: ARTIFACTS(AF) B
8: ARTIFACTS(AF) I
9: DATE(DT) B
10: DATE(DT) I
11: TIME(TI) B
12: TIME(TI) I
13 CIVILIZATION(CV) B
14: CIVILIZATION(CV) I
15: ANIMAL(AM) B
16: ANIMAL(AM) I
17: PLANT(PT) B
18: PLANT(PT) I
19: QUANTITY(QT) B
20: QUANTITY(QT) I
21: STUDY_FIELD(FD) B
22: STUDY_FIELD(FD) I
23: THEORY(TR) B
24: THEORY(TR) I
25: EVENTY(EV) B
26: EVENT(EV) I
27: MATERIAL(MT) B
28: MATERIAL(MT) I
29: TERM(TM) B
30: TERM(TM) I
'''


entity_dict = {'PS': 2, 'LC': 4, 'OG': 6, 'AF': 8, 'DT': 10, 'TI': 12, 'CV': 14, 'AM': 16, 'PT': 18, 'QT': 20,
               'FD': 22, 'TR': 24, 'EV': 26, 'MT': 28, 'TM': 30}

with open('./itos_cased.txt', encoding='utf-8') as f:
    itos = [_.replace('\n', '').replace('\r', '') for _ in f]
    stoi = {_: i for i, _ in enumerate(itos)}

missing = set()

with open('data/val_emjeol_input_cased.jsonl', 'w', encoding='utf-8') as f, \
                open('data/val_bio_just_entity_or_not.jsonl', 'w', encoding='utf-8') as f2, \
                open('data/val_bio.jsonl', 'w', encoding='utf-8') as f3:
    for fn in os.listdir('raw_data/Validation'):
        with open('./raw_data/Validation/%s' % fn, encoding='utf-8') as ff:
            datas = json.load(ff)
            for data in datas['data']:
                if 'sentence' in data:
                    if len(data['sentence']) > 1:
                        print('ddd')
                        exit()
                    else:
                        data = data['sentence'][0]
                tokens = []
                for _ in data['text']:
                    try:
                        tokens.append(stoi[_])
                    except KeyError:
                        # print(_.encode())
                        missing.add(_)
                        tokens.append(1)

                bio_jen = [0 for _ in range(len(data['text']))]
                bio = [0 for _ in range(len(data['text']))]
                try:
                    for entity in data['NE']:
                        if data['text'][entity['begin']:entity['end']] != entity['entity']:
                            entity['end'] += 1
                        if data['text'][entity['begin']:entity['end']] != entity['entity']:
                            raise KeyError
                        for idx in range(entity['begin'], entity['end']):
                            bio_jen[idx] = 2
                            bio[idx] = entity_dict[entity['type'][-2:].upper()]
                        bio_jen[entity['begin']] = 1
                        bio[entity['begin']] = entity_dict[entity['type'][-2:].upper()] - 1
                    f.write(json.dumps(tokens) + '\n')
                    f2.write(json.dumps(bio_jen) + '\n')
                    f3.write(json.dumps(bio) + '\n')
                except KeyError:
                    print(entity['type'])
for _ in missing:
    print(_)
# with open('itos_cased.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(['[PAD]', '[UNK]'] + list(token)))