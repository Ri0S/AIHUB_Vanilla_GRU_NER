import json
import os

'''
bio_just_entity_or_not.txt:
0: PAD
1: O
2: I
3: B

bio.txt:
0: PAD
1: O
2: PERSON(PS) B
3: PERSON(PS) I
4: LOCATION(LC) B
5: LOCATION(LC) I
6: ORGANIZATION(OG) B
7: ORGANIZATION(OG) I
8: ARTIFACTS(AF) B
9: ARTIFACTS(AF) I
10: DATE(DT) B
11: DATE(DT) I
12: TIME(TI) B
13: TIME(TI) I
14 CIVILIZATION(CV) B
15: CIVILIZATION(CV) I
16: ANIMAL(AM) B
17: ANIMAL(AM) I
18: PLANT(PT) B
19: PLANT(PT) I
20: QUANTITY(QT) B
21: QUANTITY(QT) I
22: STUDY_FIELD(FD) B
23: STUDY_FIELD(FD) I
24: THEORY(TR) B
25: THEORY(TR) I
26: EVENTY(EV) B
27: EVENT(EV) I
28: MATERIAL(MT) B
29: MATERIAL(MT) I
30: TERM(TM) B
31: TERM(TM) I
'''


entity_dict = {'PS': 3, 'LC': 5, 'OG': 7, 'AF': 9, 'DT': 11, 'TI': 13, 'CV': 15, 'AM': 17, 'PT': 19, 'QT': 21,
               'FD': 23, 'TR': 25, 'EV': 27, 'MT': 29, 'TM': 31}

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

                bio_jen = [1 for _ in range(len(data['text']))]
                bio = [1 for _ in range(len(data['text']))]
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