from __future__ import absolute_import, division, print_function

import os
import re
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm, trange
from fdsa import FDSA
from data_utils import RE_Processor
from sklearn.metrics import classification_report
from model import BertConfig, BertTokenizer, BertAdam, VOCAB_NAME
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def semeval_official_eval(label_map, preds, labels, outdir="./"):
    proposed_answer = os.path.join(outdir, "proposed_answer.txt")
    answer_key = os.path.join(outdir, "answer_key.txt")
    with open(proposed_answer, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(idx, label_map[pred]))
    with open(answer_key, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(labels):
            f.write("{}\t{}\n".format(idx, label_map[pred]))
    eval_cmd = "perl ./eval/semeval2010_task8_scorer-v1.2.pl {} {}".format(proposed_answer, answer_key)
    print(eval_cmd)
    p, r, f1 = 0, 0, 0
    try:
        msg = [s for s in os.popen(eval_cmd).read().split("\n") if len(s) > 0]
        b_official = False
        for i, s in enumerate(msg):
            if "(9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL" in s:
                b_official = True
            if b_official is False:
                continue
            if "MACRO-averaged result (excluding Other)" in s and "F1 =" in msg[i + 1]:
                p = float(re.findall('P = (.+?)%', msg[i + 1])[0])
                r = float(re.findall('R = (.+?)%', msg[i + 1])[0])
                f1 = float(re.findall('F1 = (.+?)%', msg[i + 1])[0])
                break
    except Exception as e:
        print(str(e))
        f1 = 0
    print("p: {}, r: {}, f1: {}".format(p, r, f1))


def train(args, model, tokenizer, processor, device):
    n_gpu = torch.cuda.device_count()
    if args.dev:
        train_examples = processor.get_train_examples(args.data_dir)
    else:
        train_examples = processor.get_train_dev_examples(args.data_dir)
    train_steps = (1 + (len(train_examples) - 1) // args.train_batch_size // args.grad_acc) * args.num_epochs
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    ogp = [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    optimizer = BertAdam(ogp, lr=args.lr, warmup=args.warmup, t_total=train_steps)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    train_data = processor.build_dataset(train_examples, tokenizer, args.max_length, mode="1")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    model.train()
    for i in trange(int(args.num_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, e1_mask, e2_mask = batch
            loss1, loss2, concat = model(input_ids, attention_mask=input_mask, labels=label_ids,
                                         e1_mask=e1_mask, e2_mask=e2_mask)
            if n_gpu > 1:
                loss1, loss2, concat = map(lambda x: x.mean(), [loss1, loss2, concat])
            train_iter.set_postfix_str(f"L1: {loss1:.4f} L2: {loss2:.4f} C: {concat:.4f}")
            loss = loss1 + loss2 * args.beta + concat
            loss = loss / args.grad_acc
            loss.backward()
            if (step + 1) % args.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
        if args.dev:
            evaluate(args, model, tokenizer, processor, device)
        evaluate(args, model, tokenizer, processor, device, "test")


def evaluate(args, model, tokenizer, processor, device, mode="dev"):
    if mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    eval_data = processor.build_dataset(examples, tokenizer, args.max_length)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating on {}".format(mode)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids, e1_mask, e2_mask = batch
        with torch.no_grad():
            logits = model.test(input_ids, attention_mask=input_mask, e1_mask=e1_mask, e2_mask=e2_mask)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
    if args.task_name == 'semeval':
        id2label_map = {i: label for label, i in processor.labels_dict.items()}
        semeval_official_eval(id2label_map, preds, out_label_ids)
    else:
        target = [i for label, i in processor.labels_dict.items() if label not in ["NA", "Other", "no_relation"]]
        result = classification_report(out_label_ids, preds, labels=target, output_dict=True)
        print("f1: {}".format(result["micro avg"]["f1-score"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", default=1e-2, type=float)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--dev", action='store_true')
    parser.add_argument("--eval_batch_size", default=20, type=int)
    parser.add_argument("--grad_acc", default=2, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--task_name", default="semeval", type=str)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--train_batch_size", default=20, type=int)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument("--warmup", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    processor = RE_Processor()
    processor.prepare_labels_dict(args.data_dir)
    label_list = processor.labels_dict.keys()
    num_labels = len(label_list)
    if args.vocab_file is None:
        args.vocab_file = os.path.join(args.model_path, VOCAB_NAME)
    print("LOAD tokenizer from", args.vocab_file)
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=True, max_len=args.max_length)
    tokenizer.add_never_split_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
    print("LOAD CHECKPOINT from", args.model_path)
    config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))
    config.__dict__["num_labels"] = num_labels
    config.__dict__["max_length"] = args.max_length
    model = FDSA.from_pretrained(args.model_path, config=config).to(args.device)
    if args.test:
        evaluate(args, model, tokenizer, processor, args.device, "test")
    else:
        train(args, model, tokenizer, processor, args.device)
