import copy
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from model import Bert_Base, Bert_Attention, Bert_LSTM


def evaluate(loader, net, device, model_name):
    """ Evaluates a model and returns loss, accuracy

    Arguments:
    loader (DataLoader): dataloader to evaluate
    net (nn.Module): Model to evaluate
    device (torch.device): Device type
    model_name (str): Model name

    """
    net.eval()

    with torch.no_grad():
        loss = 0.0
        total = 0.0
        acc = 0.0
        y_pred = []
        y_true = []
        for input_id, attention_masks, token_ids, labels in loader:
            input_id = input_id.to(device)
            attention_masks = attention_masks.to(device)
            token_ids = token_ids.to(device)
            labels = labels.long().to(device)

            if(model_name == 'base'):
                curloss, output = net(input_id, attention_masks, token_ids, labels)# noqa
            else:
                output = net(input_id, attention_masks, token_ids)
                curloss = F.cross_entropy(output, labels, reduction='sum')
            loss += curloss.item()
            preds = torch.argmax(output, 1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            acc += torch.sum(preds == labels).item()
            total += input_id.size(0)

        F1 = round((f1_score(y_true, y_pred, average='macro')), 2) * 100
        return round((loss / total), 3), round(((acc / total) * 100), 2), F1


def train_model(train_loader, dev_loader, test_loader, model_name,
                numclasses, numepochs, runs, device):
    """ Trains the neural network

    Arguments:
    train_loader (DataLoader): Training Data Loader
    dev_loader (DataLoader): Validation Data Loader
    test_loader (DataLoader): Test Data Loader
    model_name (str): Name of the model to train
    numclasses (int): Number of classes in the data
    numepochs (int): Number of epochs to train
    runs (int): Number of runs to report averaged results
    device (torch.device): Device type

    """
    avg_testacc = 0.0
    avg_testf1 = 0.0
    for run in range(1, runs+1):
        print("Training for run {} ".format(run))
        print("--------------------------------------------")
        if(model_name == 'lstm'):
            model = Bert_LSTM(numclasses).to(device)
        elif(model_name == 'attention'):
            model = Bert_Attention(numclasses, device).to(device)
        else:
            model = Bert_Base(numclasses).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        model.train()
        valbest = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(1, numepochs+1):
            model.train()
            for input_id, attention_masks, token_ids, labels in train_loader:
                input_id = input_id.to(device)
                attention_masks = attention_masks.to(device)
                token_ids = token_ids.to(device)
                labels = labels.long().to(device)

                model.zero_grad()

                if(model_name == 'base'):
                    loss, _ = model(input_id, attention_masks, token_ids, labels) # noqa
                else:
                    output = model(input_id, attention_masks, token_ids)
                    loss = F.cross_entropy(output, labels)

                loss.backward()
                optimizer.step()

            valloss, valacc, _ = evaluate(dev_loader, model, device, model_name) # noqa
            if(valacc > valbest):
                valbest = valacc
                best_model_wts = copy.deepcopy(model.state_dict())

            print("Epoch {} Val Loss {} Val Acc {} ".format(epoch,
                                                            valloss,
                                                            valacc))

        model.load_state_dict(best_model_wts)

        curtestloss, curtestacc, curtestf1 = evaluate(test_loader,
                                                      model, device,
                                                      model_name)

        print("Run {} Test Accuracy {} F1 Score {}".format(run,
                                                           curtestacc,
                                                           curtestf1))
        print("---------------------------------------------------")
        avg_testacc += curtestacc
        avg_testf1 += curtestf1

    print("Average Test Accuracy: {} F1: {} ".format(avg_testacc/runs,
                                                     avg_testf1/runs))
