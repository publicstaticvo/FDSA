import math
import random

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize, softplus
from model.bert import BertPreTrainedModel, BertModel


class FDSA(BertPreTrainedModel):
    def __init__(self, config):
        super(FDSA, self).__init__(config)
        self.bert = BertModel(config)
        self.mode = config.loss_mode
        self.normal = math.sqrt(config.hidden_size * 3) if config.normal else 1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.generator = nn.Linear(config.hidden_size * 3, config.hidden_size * 3, bias=False)
        self.ir_generator = nn.Linear(config.hidden_size * 3, config.hidden_size * 3, bias=False)
        self.concat_classifier = nn.Linear(config.hidden_size * 6, config.num_labels)
        self.apply(self.init_bert_weights)

    def extract_entity(self, sequence, e_mask):
        entity_output = torch.max(sequence.masked_fill(e_mask.bool().unsqueeze(-1), -10000), -2)[0]
        return entity_output.type_as(sequence)

    def forward(self, input_ids, attention_mask, labels, e1_mask=None, e2_mask=None):
        bs = input_ids.shape[0]
        max_length = self.config.max_length
        input_ids, attention_mask, e1_mask, e2_mask = map(lambda x: x.view(-1, max_length),
                                                          [input_ids, attention_mask, e1_mask, e2_mask])
        last_layer, pooled, _ = self.bert(input_ids, attention_mask=attention_mask)  # 4B, L, H; 4B, H
        e1_h = self.extract_entity(last_layer, e1_mask)  # 4B, H
        e2_h = self.extract_entity(last_layer, e2_mask)  # 4B, H
        feature = self.dropout(torch.cat([pooled, e1_h, e2_h], dim=-1))  # 4B, 3H
        class_re = self.generator(feature)  # 4B, 3H
        class_irre = feature - class_re
        feature, class_re, class_irre = map(lambda x: x.view(bs, 4, -1), [feature, class_re, class_irre])
        class_re = class_re[:, 0:1]  # B, 1, 3H
        logits = self.classifier(class_re).view(-1, self.config.num_labels)
        loss_fct = CrossEntropyLoss()
        loss1 = loss_fct(logits, labels.view(-1))
        anchor = class_irre[:, 0:1]  # B, 1, 3H
        positive = class_irre[:, 1:]  # B, 3, 3H
        dist_positive = torch.sum(torch.exp(-torch.norm(anchor - positive, dim=-1) / self.normal), dim=1)
        dist_negative = torch.exp(-torch.norm(anchor - anchor.transpose(0, 1), dim=-1) / self.normal) - \
            torch.eye(bs).to(anchor.device) + torch.diag(dist_positive)
        loss2 = torch.mean(-torch.log(dist_positive / torch.sum(dist_negative, dim=1)))
        class_irre.requires_grad_(True)
        dl2_dz = normalize(torch.autograd.grad(loss2, class_irre, retain_graph=True)[0], dim=-1)  # B, 4, 3H
        augment = class_irre + random.random() * dl2_dz
        concat = torch.cat([class_re.repeat(1, 4, 1), augment], dim=-1)  # B, 4, 6H
        labels = labels.unsqueeze(1).repeat(1, 4).view(-1)
        concat_loss = loss_fct(self.concat_classifier(concat.view(bs * 4, -1)), labels)
        return loss1, loss2, concat_loss

    def test(self, input_ids, attention_mask, e1_mask=None, e2_mask=None):
        last_layer, pooled, _ = self.bert(input_ids, attention_mask=attention_mask)
        e1_h = self.extract_entity(last_layer, e1_mask)
        e2_h = self.extract_entity(last_layer, e2_mask)
        feature = self.dropout(torch.cat([pooled, e1_h, e2_h], dim=-1))
        class_re = self.generator(feature)
        logits = self.classifier(class_re).view(-1, self.config.num_labels)
        return torch.argmax(logits, dim=-1)
