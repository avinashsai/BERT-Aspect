import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification

fchidden = 256
hiddendim_lstm = 128
embeddim = 768
numlayers = 12


class Bert_Base(nn.Module):
    def __init__(self, numclasses):
        super(Bert_Base, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.dropout = nn.Dropout(0.1)

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', # noqa
                                                                   output_hidden_states=False, # noqa
                                                                   output_attentions=False, # noqa
                                                                   num_labels=self.numclasses) # noqa
        print("BERT Model Loaded")

    def forward(self, inp_ids, att_mask, token_ids, labels):
        loss, out = self.bert(input_ids=inp_ids, attention_mask=att_mask,
                              token_type_ids=token_ids, labels=labels)
        return loss, out


class Bert_LSTM(nn.Module):
    def __init__(self, numclasses):
        super(Bert_LSTM, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.hiddendim_lstm = hiddendim_lstm
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT Model Loaded")
        self.lstm = nn.LSTM(self.embeddim, self.hiddendim_lstm, batch_first=True) # noqa
        self.fc = nn.Linear(self.hiddendim_lstm, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class Bert_Attention(nn.Module):
    def __init__(self, numclasses, device):
        super(Bert_Attention, self).__init__()
        self.numclasses = numclasses
        self.embeddim = embeddim
        self.numlayers = numlayers
        self.fchidden = fchidden
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT Model Loaded")

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embeddim))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embeddim, self.fchidden)) # noqa
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(device)

        self.fc = nn.Linear(self.fchidden, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(0, self.numlayers)], dim=-1) # noqa
        hidden_states = hidden_states.view(-1, self.numlayers, self.embeddim)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v
