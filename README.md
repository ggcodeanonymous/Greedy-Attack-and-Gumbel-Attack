# Greedy Attack and Gumbel Attack: Generating Adversarial Examples for Discrete Data

Code for replicating the experiments.

## Dependencies
The code runs with Python 2.7 and requires Tensorflow 1.2.1, Keras 2.1.5 and nltk 3.2.5. Please `pip install` the following packages:
- `numpy`
- `pandas`
- `tensorflow` 
- `nltk`
- `keras`

## AG's News data set with a trained model
For ease of replication, the AG's News data set and a trained character-level convolutional networks (Char-CNN) are provided. Please cite [Character-level convolutional networks for text classification](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica) for the data and the model. 

The AG's News data set can be downloaded [here](https://drive.google.com/open?id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms). Please put the folder in the directory agccnn/data/ and unzip it. The weights of a trained character-level convolutional networks (Char-CNN) can be downloaded [here](https://drive.google.com/open?id=1dkt_sfRPJzQO3uMA4afcSRgNMFfOrk3M). Please put the file in the directory agccnn/params/.

## Generation of Greedy Attack adversarial examples
Generate the Greedy Attack adversarial examples for Char-CNN on AG's News.

First stage: search for the most important k features.
```shell
python score.py --data agccnn  --method leave_one_out --num_feats 10
```
Second stage: search for values to replace the selected k features.
```shell
python change.py --data agccnn --method leave_one_out --num_feats 10 --changing_way greedy_change_k 
```

## Generation of Gumbel Attack adversarial examples
Generate the Gumbel Attack adversarial examples for Char-CNN on AG's News.

Generate predictions for training of Gumbel Attack.
```shell
python score.py --data agccnn --method create_predictions
```

Train first-stage Gumbel Attack. 
```shell
python score.py --data agccnn --method L2X --num_feats 5 --original --mask --train
```

Apply first-stage Gumbel Attack on both training and test sets.
```shell
python score.py --data agccnn --method L2X --num_feats 5 --original --mask --train_score
python score.py --data agccnn --method L2X --num_feats 5 --original --mask 
```

Train second-stage Gumbel Attack.
```shell
python change.py --data agccnn --method L2X --num_feats 5 --original --mask --changing_way gumbel --train
```

Create adversarial examples.
```shell
python change.py --data agccnn --method L2X --num_feats 5 --original --mask --changing_way gumbel
```
