import torch
import torch.nn as nn
from transformers import BertModel


class Bert_Aspect(nn.Module):
    def __init__(self, numclasses, hiddendim=128):
        super(Bert_Aspect, self).__init__()
        self.numclasses = numclasses
        self.hiddendim = hiddendim
        self.dropout = nn.Dropout(0.1)

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=False)
        print("BERT Model Loaded")
        self.lstm = nn.LSTM(768, self.hiddendim, batch_first=True)
        self.fc = nn.Linear(self.hiddendim, self.numclasses)

    def forward(self, inp_ids, att_mask, token_ids):
        last_hidden_state, pooler_output, \
                hidden_states = self.bert(input_ids=inp_ids,
                                          attention_mask=att_mask,
                                          token_type_ids=token_ids)

        hidden_states = torch.stack([hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(12)], dim=-1)
        hidden_states = hidden_states.view(-1, 12, 768)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
