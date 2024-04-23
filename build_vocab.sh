#!/bin/bash
# build vocab for different datasets
dataset_dir="dataset"
python prepare_vocab.py --data_dir $dataset_dir/Restaurants --vocab_dir $dataset_dir/Restaurants
python prepare_vocab.py --data_dir $dataset_dir/Laptops --vocab_dir $dataset_dir/Laptops
python prepare_vocab.py --data_dir $dataset_dir/Tweets --vocab_dir $dataset_dir/Tweets
python prepare_vocab.py --data_dir $dataset_dir/MAMS --vocab_dir $dataset_dir/MAMS
