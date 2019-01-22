# Greedy-Attack-and-Gumbel-Attack: Generating Adversarial Examples for Discrete Data

Code for replicating the experiments.

## Dependencies
The code runs with Python 2.7 and requires Tensorflow 1.2.1, Keras 2.1.5 and nltk 3.2.5. Please `pip install` the following packages:
- `numpy`
- `pandas`
- `tensorflow` 
- `nltk`
- `keras`

## Generation of Greedy Attack adversarial examples
Before generating adversarial samples, we first train a word-based CNN model for IMDB data set.
```shell
python score.py --data imdbcnn  --method training
```
Generate the Greedy Attack adversarial examples for IMDB with the word-based CNN model.
```shell
python score.py --data imdbcnn  --method leave_one_out --num_feats 10
python change.py --data imdbcnn --method leave_one_out --num_feats 10 --changing_way greedy_change_k 
```

## Generation of Gumbel Attack adversarial examples
Generate the Gumbel Attack adversarial examples for IMDB data set with the word-based CNN model.
```shell
python score.py --data imdbcnn --method create_predictions
python score.py --data imdbcnn --method L2X --num_feats 5 --original --mask --train
python score.py --data imdbcnn --method L2X --num_feats 5 --original --mask 
python score.py --data imdbcnn --method L2X --num_feats 5 --original --mask --train_score
python change.py --data imdbcnn --method L2X --num_feats 5 --original --mask --changing_way gumbel --train
python change.py --data imdbcnn --method L2X --num_feats 5 --original --mask --changing_way gumbel
```

## Other data sets
To generate adversarial examples for AG’s News corpus with a character-based CNN, replace 'imdbcnn' with 'agccnn' and add '--max_words 69' as there are 69 characters in total.

To generate adversarial examples for Yahoo! Answers with an LSTM, replace 'imdbcnn' with 'yahoolstm' respectively. 
