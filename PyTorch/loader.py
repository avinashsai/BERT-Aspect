from ast import literal_eval
import torch.utils.data as utils


def load_data(dataset, num_classes, data_path=None):

    label = {'negative': 0,
             'positive': 1,
             'neutral': 2,
             'conflict': 3}

    if(data_path is None):
        data_path = '../../Data/'
    train_file = data_path + 'atsa-' + dataset + '/atsa_train.json'
    test_file = data_path + 'atsa-' + dataset + '/atsa_test.json'

    temp = open(train_file, 'r', encoding='latin1').read()
    train = literal_eval(temp)
    train_sentence = []
    train_aspect = []
    train_sentiment = []
    for xml in train:
        if(xml['sentiment'] == 'conflict' and num_classes == 3):
            continue
        train_sentence.append(xml['sentence'])
        train_aspect.append(xml['aspect'])
        train_sentiment.append(label[xml['sentiment']])

    temp = open(test_file, 'r', encoding='latin1').read()
    test = literal_eval(temp)
    test_sentence = []
    test_aspect = []
    test_sentiment = []
    for xml in test:
        if(xml['sentiment'] == 'conflict' and num_classes == 3):
            continue
        test_sentence.append(xml['sentence'])
        test_aspect.append(xml['aspect'])
        test_sentiment.append(label[xml['sentiment']])

    train_sen_len = [len(sentence.split()) for sentence in train_sentence]
    train_asp_len = [len(aspect.split()) for aspect in train_aspect]

    print("----------------------------------------")
    print("Maximum Training data Sentence Length: {}".format(
                                                       max(train_sen_len)))
    print("Maximum Training data Aspect Length: {}".format(max(train_asp_len)))
    print("----------------------------------------")

    return (train_sentence, train_aspect, train_sentiment, test_sentence,
            test_aspect, test_sentiment)


def get_loader(input_ids, attention_masks, token_type_ids, labels, batchsize):
    array = utils.TensorDataset(input_ids, attention_masks, token_type_ids,
                                labels)
    loader = utils.DataLoader(array, batch_size=batchsize)
    return loader
