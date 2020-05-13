import torch
from transformers import BertTokenizer


def get_pretrained_tokenizer():

    print("Downloading bert tokenizer to cache")
    print("---------------------------------------")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                   do_lower_case=True
                                                   )
    return bert_tokenizer


def tokenize_sentences(bert_tokenizer, sentences, aspects, maxlen):

    input_ids = []
    attention_masks = []
    token_type_ids = []

    for sentence, aspect in zip(sentences, aspects):

        encoded = bert_tokenizer.encode_plus(text=sentence,
                                             text_pair=aspect,
                                             add_special_tokens=True,
                                             max_length=maxlen,
                                             pad_to_max_length=True,
                                             return_attention_masks=True,
                                             return_token_type_ids=True,
                                             return_tensors='pt')

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0, out=None)
    attention_masks = torch.cat(attention_masks, dim=0, out=None)
    token_type_ids = torch.cat(token_type_ids, dim=0, out=None)

    return input_ids, attention_masks, token_type_ids
