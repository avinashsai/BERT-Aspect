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
| BERT Base Uncased + Linear | 75.44 |74.66  |
| BERT Base Uncased + LSTM   | 76    | 75.31  |
| BERT Base Uncased + Attention | 75.91 |75.16|

| Model | This Implementation Result (F1) | Paper Result (F1) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + Linear | 70 | 68.64 |
| BERT Base Uncased + LSTM   |  70.6  |  69.37 |
| BERT Base Uncased + Attention | 70.6 | 68.76 |

For Restaurant dataset:

| Model | This Implementation Result (Acc) | Paper Result (Acc) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + Linear | 82.91 | 81.92 |
| BERT Base Uncased + LSTM   |   83.04 | 82.21  |
| BERT Base Uncased + Attention | 83.29 | 82.38 |

| Model | This Implementation Result (F1) | Paper Result (F1) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + Linear | 73.2 | 71.97 |
| BERT Base Uncased + LSTM   |  73.4  | 72.52  |
| BERT Base Uncased + Attention | 73.6 | 73.22 |

For Twitter dataset:

| Model | This Implementation Result (Acc) | Paper Result (Acc) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + Linear | 70.32 | 72.46 |
| BERT Base Uncased + LSTM   |  70.66  | 73.06  |
| BERT Base Uncased + Attention |69.06  | 73.35 |

| Model | This Implementation Result (F1) | Paper Result (F1) | 
|-------|----------------------------------|--------------------|
| BERT Base Uncased + Linear | 68.5 | 71.04 |
| BERT Base Uncased + LSTM   | 67.1   | 71.61  |
| BERT Base Uncased + Attention |  69|71.88  |
