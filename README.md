# BERT-Aspect
Code for the paper "Utilizing BERT Intermediate Layers for Aspect Based Sentiment Analysis and Natural Language Inference" https://arxiv.org/pdf/2002.04815.pdf.

# Requirements
```
python>=3.6
transformers==2.9.0
pytorch==1.5.0
```
# Running the code
```
git clone https://github.com/avinashsai/BERT-Aspect.git
cd PyTorch
```
```
python main.py --dataset (laptop/ restaurant)
               --maxlen (Maximum Sentence length (default: 80))
               --numclasses (3 if "conflict" class is not included else 4 (default:3))
               --data-path (path to datasets (default: '../Data/))
               --batch-size (Batch Size (default: 8)
               --numepochs (Number of training epochs (default: 10))
               --runs (Number of average runs to report results (default: 10))
               --model_name (lstm /attention /base)
```
# Note
This code is un-official implementation of the paper. Hence, training details may not be exactly similar. Also, I have made couple of changes due to which results are superior than the reported paper results.

# Results

For Laptop dataset:

| Model | This Implementation Result (Acc) | Paper Result (Acc) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + LSTM | 76.03 | 75.31 |

| Model | This Implementation Result (F1) | Paper Result (F1) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + LSTM | 70.9 | 69.37 |

For Restaurant dataset:

| Model | This Implementation Result (Acc) | Paper Result (Acc) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + LSTM | 83.6 | 82.21 |

| Model | This Implementation Result (F1) | Paper Result (F1) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + LSTM | 74.5 | 73.22 |
