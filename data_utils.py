import os
import copy
import json
import torch
import random
from torch.utils.data import Dataset
from collections import defaultdict


def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word


def pad(x, max_length):
    if len(x) > max_length:
        if x[max_length] != 0:
            x[max_length - 1] = 102
        return x[:max_length]
    return x + [0 for _ in range(max_length - len(x))]


def position(i, s, m):
    try:
        return i.index(s)
    except ValueError:
        return m


def getmask(input_id, special_tokens, max_length):
    try:
        seq_length = min(input_id.index(0), max_length)
    except ValueError:
        seq_length = max_length
    input_mask = [1 for _ in range(seq_length)] + [0 for _ in range(max_length - seq_length)]
    input_id = pad(input_id, max_length)
    e11, e12, e21, e22 = map(lambda s: position(input_id, s, max_length - 2), special_tokens)
    e1_mask = [0 for _ in range(e11)] + [1 for _ in range(e11, e12 + 1)] + [0 for _ in range(max_length - e12 - 1)]
    e2_mask = [0 for _ in range(e21)] + [1 for _ in range(e21, e22 + 1)] + [0 for _ in range(max_length - e22 - 1)]
    return input_id, input_mask, e1_mask, e2_mask


def augment(feature, special_tokens, max_length, e_set=None, e2_set=None):
    input_id = feature["input_ids"]
    self_label = feature["label_id"]
    e11, e12, e21, e22 = (input_id.index(s) for s in special_tokens)
    e11m, e12m, e21m, e22m = min(e11, e21), min(e12, e22), max(e11, e21), max(e12, e22)
    input_id1 = input_id[:e11m] + input_id[e21m:e22m + 1] + input_id[e12m + 1:e21m] + input_id[e11m:e12m + 1] + input_id[e22m + 1:]
    input_id2 = input_id[:e11m + 1] + input_id[e12m:e21m + 1] + input_id[e22m:]
    selectable_e1, selectable_e2 = [], []
    if random.random() < 0.5:
        selectable_e1, selectable_e2 = e_set[self_label], e2_set[self_label]
    else:
        for i in range(len(e_set)):
            selectable_e1 += e_set[i]
            selectable_e2 += e2_set[i]
    randi = random.randint(0, len(selectable_e1) - 1)
    randj = random.randint(0, len(selectable_e2) - 1)
    new_e1 = selectable_e1[randi]
    new_e2 = selectable_e2[randj]
    if e11m == e21:
        new_e1, new_e2 = new_e2, new_e1
    input_id3 = input_id[:e11m + 1] + new_e1 + input_id[e12m:e21m + 1] + new_e2 + input_id[e22m:]
    input_id1, input_mask1, e1_mask1, e2_mask1 = getmask(input_id1, special_tokens, max_length)
    input_id2, input_mask2, e1_mask2, e2_mask2 = getmask(input_id2, special_tokens, max_length)
    input_id3, input_mask3, e1_mask3, e2_mask3 = getmask(input_id3, special_tokens, max_length)
    return [input_id1, input_id2, input_id3], [input_mask1, input_mask2, input_mask3], \
        [e1_mask1, e1_mask2, e1_mask3], [e2_mask1, e2_mask2, e2_mask3]


class REDataset(Dataset):
    def __init__(self, features, max_seq_length, mode, e1_list, e2_list, special):
        self.ml = max_seq_length
        self.data = features
        self.e1s = e1_list
        self.e2s = e2_list
        self.spc = special
        self.mode = mode

    def __getitem__(self, index):
        item = copy.deepcopy(self.data[index])
        if self.mode != '0':
            input_id, input_mask, e1_mask, e2_mask = augment(item, self.spc, self.ml, self.e1s, self.e2s)
            item["input_ids"] = [item['input_ids']] + input_id
            item["input_mask"] = [item['input_mask']] + input_mask
            item["e1_mask"] = [item['e1_mask']] + e1_mask
            item["e2_mask"] = [item['e2_mask']] + e2_mask
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        input_mask = torch.tensor(item["input_mask"], dtype=torch.long)
        e1_mask_ids = torch.tensor(item["e1_mask"], dtype=torch.float)
        e2_mask_ids = torch.tensor(item["e2_mask"], dtype=torch.float)
        label_ids = torch.tensor(item["label_id"], dtype=torch.long)
        return input_ids, input_mask, label_ids, e1_mask_ids, e2_mask_ids  # input size: BS * 4 * L

    def __len__(self):
        return len(self.data)


class RE_Processor:
    def __init__(self, direct=True, labels_dict=None):
        if labels_dict is None:
            labels_dict = {}
        self.direct = direct
        self.labels_dict = labels_dict

    def get_train_examples(self, data_dir):
        return self._create_examples(self.get_knowledge_feature(data_dir, flag="train"), "train")

    def get_train_dev_examples(self, data_dir):
        return self._create_examples(self.get_knowledge_feature(data_dir, flag="train") +
                                     self.get_knowledge_feature(data_dir, flag="dev"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.get_knowledge_feature(data_dir, flag="dev"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self.get_knowledge_feature(data_dir, flag="test"), "test")

    def get_knowledge_feature(self, data_dir, flag="train"):
        return self.read_features(data_dir, flag=flag)

    def get_labels(self, data_dir):
        label_path = os.path.join(data_dir, "label.json")
        if not os.path.exists(label_path):
            labels = set()
            for flag in ["train", "test", "dev"]:
                datafile = os.path.join(data_dir, '{}.txt'.format(flag))
                if os.path.exists(datafile) is False:
                    continue
                all_data = self.load_textfile(datafile)
                for data in all_data:
                    labels.add(data['label'])
            labels = sorted(list(labels))
            with open(label_path, 'w') as f:
                json.dump(labels, f)
        else:
            labels = json.load(open(label_path, 'r'))
        return labels

    def get_key_list(self):
        return self.keys_dict.keys()

    def _create_examples(self, features, set_type):
        examples = []
        for i, feature in enumerate(features):
            guid = "%s-%s" % (set_type, i)
            feature["guid"] = guid
            examples.append(feature)
        return examples

    def prepare_keys_dict(self, data_dir):
        keys_frequency_dict = defaultdict(int)
        for flag in ["train", "test", "dev"]:
            datafile = os.path.join(data_dir, '{}.txt'.format(flag))
            if not os.path.exists(datafile):
                continue
            all_data = self.load_textfile(datafile)
            for data in all_data:
                for word in data['words']:
                    keys_frequency_dict[change_word(word)] += 1
        keys_dict = {"[UNK]": 0}
        for key, freq in sorted(keys_frequency_dict.items(), key=lambda x: x[1], reverse=True):
            keys_dict[key] = len(keys_dict)
        self.keys_dict = keys_dict
        self.keys_frequency_dict = keys_frequency_dict

    def prepare_labels_dict(self, data_dir):
        label_list = self.get_labels(data_dir)
        labels_dict = {}
        for label in label_list:
            labels_dict[label] = len(labels_dict)
        self.labels_dict = labels_dict

    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def read_features(self, data_dir, flag):
        all_text_data = self.load_textfile(os.path.join(data_dir, '{}.txt'.format(flag)))
        all_feature_data = []
        for text_data in all_text_data:
            label = text_data["label"]
            if label == "other":
                label = "Other"
            ori_sentence = text_data["ori_sentence"].split(" ")
            all_feature_data.append({
                "ori_sentence": ori_sentence,
                "label": label,
                "e1": text_data["e1"],
                "e2": text_data["e2"],
            })

        return all_feature_data

    def load_textfile(self, filename):
        data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    items = line.strip().split("\t")
                    if len(items) != 4:
                        continue
                    e1, e2, label, sentence = items
                    data.append({
                        "e1": e1,
                        "e2": e2,
                        "label": label,
                        "ori_sentence": sentence,
                    })
        return data

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = self.labels_dict
        features = []
        special = tokenizer.convert_tokens_to_ids(["<e1>", "</e1>", "<e2>", "</e2>"])
        e1_list = [[] for _ in range(len(label_map))]
        e2_list = [[] for _ in range(len(label_map))]
        for (ex_index, example) in enumerate(examples):
            tokens = ["[CLS]"]
            e1_mask = [0]
            e2_mask = [0]
            e1_mask_val = 0
            e2_mask_val = 0
            for i, word in enumerate(example["ori_sentence"]):
                if word == "<e1>":
                    e1_mask_val = 1
                    tokens.append(word)
                    e1_mask.append(1)
                    e2_mask.append(0)
                elif word == "</e1>":
                    tokens.append(word)
                    e1_mask.append(1)
                    e2_mask.append(0)
                    e1_mask_val = 0
                elif word == "<e2>":
                    e2_mask_val = 1
                    tokens.append(word)
                    e1_mask.append(0)
                    e2_mask.append(1)
                elif word == "</e2>":
                    tokens.append(word)
                    e1_mask.append(0)
                    e2_mask.append(1)
                    e2_mask_val = 0
                else:
                    token = tokenizer.tokenize(change_word(word).lower())
                    tokens.extend(token)
                    e1_mask.extend([e1_mask_val for _ in token])
                    e2_mask.extend([e2_mask_val for _ in token])
            tokens.append("[SEP]")
            e1_mask.append(0)
            e2_mask.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_ids, input_mask, e1_mask, e2_mask = map(
                lambda x: pad(x, max_seq_length), [input_ids, input_mask, e1_mask, e2_mask])
            label_id = label_map[example["label"]]
            features.append({
                "input_ids": input_ids,
                "input_mask": input_mask,
                "label_id": label_id,
                "e1_mask": e1_mask,
                "e2_mask": e2_mask,
            })
            e1 = input_ids[input_ids.index(special[0]) + 1:input_ids.index(special[1])]
            e2 = input_ids[input_ids.index(special[2]) + 1:input_ids.index(special[3])]
            e1_list[label_id].append(e1)
            e2_list[label_id].append(e2)
        return features, e1_list, e2_list, special

    def build_dataset(self, examples, tokenizer, max_seq_length, mode="0"):
        features, e1_list, e2_list, special = self.convert_examples_to_features(examples, tokenizer, max_seq_length)
        return REDataset(features, max_seq_length, mode, e1_list, e2_list, special)
