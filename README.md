# Comparison of some neural network architectures on small data

### Data 
Folder /data

I collect several famous russian classic novels (War and peace, Anna Karenina, 
Crime and punishment etc) in txt and stacked them together. 

The corpora is approximetely 18MB.

### NN architectures
Folder /models

I tried the most popular RNN architectures: GRU and LSTM with different capacity. 

![Feature importance](/img/RNN.png)

I made simple laguage models, so the quality metric is perplexity.

### Results

| Model            | Perplexity    | Train time | 
| ---------------- |:-------------:| ----------:|
| GRU              | 5772.1        | 5 min 20s  | 
| GRU + dropout    | 2641.9        | 5min 56s   |
| LSTM             | 3915.2        | 5min 37s   | 
| LSTM + dropout   | 2765.7        | 6 min 5s   | 
| GRU Large + dropout | 4275.2     | 8min 30s   | 


Dropout have improved LM significantly. However, LSTM + dropout performs almost the same as GRU + dropout. Thus, use dropout to achieve better performance, epspecially on a small corpora.
