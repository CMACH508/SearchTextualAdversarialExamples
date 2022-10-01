# Searching for Textual Adversarial Examples with Learned Strategy.

This repository contains the codes and the resources for our paper: Searching for Textual Adversarial Examples with Learned Strategy.

## Requirements

* datasets==1.5.0
* pytorch==1.8.1
* scikit-learn==0.20.3
* nltk==3.5
* transformers==4.5.0
* sentence-transformers==1.1.0

## Resources

### Victim Models

You can download the victim models from this [link](https://pan.baidu.com/s/1qOnRGKmBb7qfrBjGWXLKZw) (access code: 8n96), and place the extracted files in `resources/victim_models`. The directory structure should be like:

```
├── resources
│   ├── victim_models
│   │   ├── bert-imdb
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   ├── bert-mr
│   │   │   └── ...
│   │   ├── bert-snli
│   │   │   └── ...
│   │   ├── bert-yelp
│   │   │   └── ...
│   │   ├── lstm-imdb
│   │   │   └── ...
│   │   ├── lstm-mr
│   │   │   └── ...
│   │   └── lstm-yelp
│   │       └── ...
│   └── ...
└── ...
```

The single-text input datasets have both BERT and LSTM as victim models, and the text-pair input datasets have only BERT as victim models.

### Pretrained Language Models

We use the pretrained BERT model in the synonym selection network, and use GPT-2 to evaluate the perplexity. The pretrained models can be downloaded from HuggingFace:

* BERT: https://huggingface.co/bert-base-uncased
* GPT-2: https://huggingface.co/gpt2

The model files should be placed in `resources/encoder_models`. The directory structure should be like:

```
├── resources
│   ├── encoder_models
│   │   ├── bert
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── vocab.txt
│   │   │   └── ...
│   │   └── gpt2
│   │       ├── config.json
│   │       ├── merges.txt
│   │       ├── pytorch_model.bin
│   │       ├── tokenizer.json
│   │       ├── vocab.json
│   │       └── ...
│   └── ...
└── ...
```

### Datasets

The original datasets are in `resources/datasets/{victim_model}_original`, where the `{victim_model}` should be replaced by `bert` or `lstm`. In our paper, in order to strictly comply with the black box setting, the training data of the synonym selection network is from other datasets. Specificly, the training data part in MR is from IMDB, the training data part in Yelp is from IMDB, the training data part in IMDB is from Yelp, the training data part of SNLI is from MNLI. 

To construct the training data for the two networks, run the following command:

```
python construct_training_data.py [--device DEVICE]
                                  [--dataset_path DATASET_PATH]
                                  [--output_path OUTPUT_PATH]
                                  [--model_path MODEL_PATH]
                                  [--gpt_path GPT_PATH]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--output_path: the generated training dataset path, e.g., resources/datasets/bert_train/imdb.json
--model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
```

For example, to generate training dataset for BERT victim model on the IMDB dataset:

```
python construct_training_data.py --device cuda --dataset_path resources/datasets/bert_original/imdb.json --output_path resources/datasets/bert_train/imdb.json --model_path resources/victim_models/bert-imdb --gpt_path resources/encoder_models/gpt2
```

This step may take a long time, and we provide generated training data in this [link](https://pan.baidu.com/s/1uk0j6zL3jtU_ZavCo39WYw) (access code: zmbb).

Finally, the directory structure should be like:

```
├── resources
│   ├── datasets
│   │   ├── bert_original
│   │   │   ├── imdb.json
│   │   │   ├── yelp.json
│   │   │   ├── mr.json
│   │   │   └── snli.json
│   │   ├── bert_train
│   │   │   ├── imdb.json
│   │   │   ├── yelp.json
│   │   │   ├── mr.json
│   │   │   └── snli.json
│   │   ├── lstm_original
│   │   │   ├── imdb.json
│   │   │   ├── yelp.json
│   │   │   └── mr.json
│   │   └── lstm_train
│   │       ├── imdb.json
│   │       ├── yelp.json
│   │       └── mr.json
│   └── ...
└── ...
```

## Running

### Training

#### Synonym Selection Network

To train the synonym selection network, run the following command:

```
python attack.py train_candidate_selection_module [--device DEVICE]
                                                  [--encoder_path ENCODER_PATH]
                                                  [--train_data_path TRAIN_DATA_PATH]
                                                  [--module_path MODULE_PATH]
                                                  [--sim_threshold SIM_THRESHOLD]
                                                  [--ppl_proportion PPL_PROPORTION]
                                                  [--hidden_dim HIDDEN_DIM]
                                                  [--num_classes NUM_CLASSES]
                                                  [--num_epochs NUM_EPOCHS]
                                                  [--batch_size BATCH_SIZE]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--encoder_path: the path of BERT model, e.g., resources/encoder_models/bert
--train_data_path: the training dataset path, e.g., resources/datasets/bert_train/imdb.json
--module_path: the path of the synonym selection network, e.g., results/prediction/imdb/bert/synonym_selection_module
--sim_threshold: the similarity threshold for training data selection, set to 0.95 in the experiment.
--ppl_proportion: the perplexity threshold for training data selection, set to 0.9 in the experiment.
--hidden_dim: the hidden dimension, set to 128 in the experiment.
--num_classes: the number of classes of the dataset, set to 2 for IMDB, Yelp and MR, set to 3 for SNLI.
--num_epochs: the number of training epochs, set to 5 in the experiment.
--batch_size: the number of batch size, set to 5 in the experiment.
```

For example, to train the synonym selection network for BERT victim model on the IMDB dataset:

```
python attack.py train_candidate_selection_module --device cuda --encoder_path resources/encoder_models/bert --train_data_path resources/datasets/bert_train/imdb.json --module_path results/prediction/imdb/bert/synonym_selection_network --sim_threshold 0.95 --ppl_proportion 0.9 --hidden_dim 128 --num_classes 2 --num_epochs 5 --batch_size 5
```

#### Pretrained Models

We have provided the pretrained models, which can be downloaded from this [link](https://pan.baidu.com/s/1EDNVSaduxCGBwfmsvSqa6w) (access code: s5nd).

When extracted, the directory structure should be like the following, and the directory name format is `{dataset}/{victim_model}`.

```
├── imdb
│   ├── bert
│   │   └── synonym_selection_network
│   │       ├── pytorch_model.bin
│   │       └── train_args.json
│   └── lstm
│       └── synonym_selection_network
│           ├── pytorch_model.bin
│           └── train_args.json
├── mr
│   └── bert
│       └── ...
├── snli
│   └── bert
│       └── ...
└── yelp
    ├── bert
    │   └── ...
    └── lstm
        └── ...
```

### Evaluation

#### Beam Search

For beam search, run the following command:

```
python attack.py test [--device DEVICE]
                      [--encoder_path ENCODER_PATH]
                      [--dataset_path DATASET_PATH]
                      [--gpt_path GPT_PATH]
                      [--hidden_dim HIDDEN_DIM]
                      [--num_classes NUM_CLASSES]
                      [--sim_threshold SIM_THRESHOLD]
                      [--search_method SEARCH_METHOD]
                      [--beam_size BEAM_SIZE]
                      [--wr_top_k WR_TOP_K]
                      [--top_k TOP_K]
                      [--victim_model_path VICTIM_MODEL_PATH]
                      [--candidate_selection_module_path CANDIDATE_SELECTION_MODULE_PATH]
                      [--output_path OUTPUT_PATH]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--encoder_path: the path of BERT model, e.g., resources/encoder_models/bert
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
--hidden_dim: the hidden dimension, set to 128 in the experiment.
--num_classes: the number of classes of the dataset, set to 2 for IMDB, Yelp and MR, set to 3 for SNLI.
--sim_threshold: the similarity threshold for generating adversarial examples, set to 0.9 in the experiment.
--search_method: the search method, use beam_search for beam search.
--beam_size: the beam size, set to 1 or 4 in the experiment.
--wr_top_k: the top_k in word ranking, set to 5 for beam seach in the experiment.
--top_k: the top_k in synonym selection, set to 15 in the experiment.
--victim_model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--candidate_selection_module_path: the path of the synonym selection network, e.g., results/prediction/imdb/bert/synonym_selection_network/pytorch_model.bin
--output_path: the output path, e.g., results/prediction/imdb/bert
```

For example, to evaluate the beam search based attack againt BERT model on the IMDB dataset:

```
python attack.py test --device cuda --encoder_path resources/encoder_models/bert --dataset_path resources/datasets/bert_original/imdb.json --gpt_path resources/encoder_models/gpt2 --hidden_dim 128 --num_classes 2 --sim_threshold 0.9 --search_method beam_search --beam_size 1 --wr_top_k 5 --top_k 15 --victim_model_path resources/victim_models/bert-imdb --candidate_selection_module_path results/prediction/imdb/bert/synonym_selection_network/pytorch_model.bin --output_path results/prediction/imdb/bert
```

#### MCTS

For MCTS, run the following command:

```
python attack.py test [--device DEVICE]
                      [--encoder_path ENCODER_PATH]
                      [--dataset_path DATASET_PATH]
                      [--gpt_path GPT_PATH]
                      [--hidden_dim HIDDEN_DIM]
                      [--num_classes NUM_CLASSES]
                      [--sim_threshold SIM_THRESHOLD]
                      [--search_method SEARCH_METHOD]
                      [--mcts_search_budget MCTS_SEARCH_BUDGET]
                      [--mcts_exploration_coefficient MCTS_EXPLORATION_COEFFICIENT]
                      [--mcts_state_value_coefficient MCTS_STATE_VALUE_COEFFICIENT]
                      [--wr_top_k WR_TOP_K]
                      [--top_k TOP_K]
                      [--victim_model_path VICTIM_MODEL_PATH]
                      [--candidate_selection_module_path CANDIDATE_SELECTION_MODULE_PATH]
                      [--output_path OUTPUT_PATH]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--encoder_path: the path of BERT model, e.g., resources/encoder_models/bert
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
--hidden_dim: the hidden dimension, set to 128 in the experiment.
--num_classes: the number of classes of the dataset, set to 2 for IMDB, Yelp and QQP, set to 3 for SNLI.
--sim_threshold: the similarity threshold for generating adversarial examples, set to 0.9 in the experiment.
--search_method: the search method, use mcts for mcts.
--mcts_search_budget: the maximum number of search iterations, set to 200 in the experiment.
--mcts_exploration_coefficient: the scaling factor for evaluating action, set to 1.0 in the experiment.
--mcts_state_value_coefficient: the scaling factor for evaluating state, set to 0.4 in the experiment.
--wr_top_k: the top_k in word ranking, set to 5 for beam seach in the experiment.
--top_k: the top_k in synonym selection, set to 3 in the experiment.
--victim_model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--candidate_selection_module_path: the path of the synonym selection network, e.g., results/prediction/imdb/bert/synonym_selection_network/pytorch_model.bin
--output_path: the output path, e.g., results/prediction/imdb/bert
```

For example, to evaluate the mcts based attack againt BERT model on the IMDB dataset:

```
python attack.py test --device cuda --encoder_path resources/encoder_models/bert --dataset_path resources/datasets/bert_original/imdb.json --gpt_path resources/encoder_models/gpt2 --hidden_dim 128 --num_classes 2 --sim_threshold 0.9 --search_method mcts --mcts_search_budget 200 --mcts_exploration_coefficient 1.0 --mcts_state_value_coefficient 0.4 --wr_top_k 5 --top_k 3 --victim_model_path resources/victim_models/bert-imdb --candidate_selection_module_path results/prediction/imdb/bert/synonym_selection_network/pytorch_model.bin --output_path results/prediction/imdb/bert
```

### Greedy

We also provide implementation for the original greedy methods, run the following command:

```
python greedy.py [--device DEVICE]
                 [--dataset_path DATASET_PATH]
                 [--model_path MODEL_PATH]
                 [--output_path OUTPUT_PATH]
                 [--gpt_path GPT_PATH]
                 [--word_order WORD_ORDER]
                 [--sim_threshold SIM_THRESHOLD]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--output_path: the output path, e.g., results/greedy/imdb/bert
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
--word_order: the method for determining word substitution order, can be text-fooler or pwws.
--sim_threshold: the similarity threshold for generating adversarial examples, set to 0.9 in the experiment.
```

For example, to generate adversarial examples in text-fooler word substitution order:

```
python greedy.py --device cuda --dataset_path resources/datasets/bert_original/imdb.json --model_path resources/victim_models/bert-imdb --output_path results/greedy/imdb/bert --word_order text-fooler --sim_threshold 0.9
```

## Acknowledgement

This repository used some codes in [TextAttack](https://github.com/QData/TextAttack).

## Citation

If you find this repo useful, please cite our paper:

```
@inproceedings{search_textual_adversarial_examples,
 title = {Searching for Textual Adversarial Examples with Learned Strategy},
 author = {Xiangzhe, Guo  and Ruidan, Su  and Shikui, Tu  and Lei, Xu},
 booktitle = {The 29th International Conference on Neural Information Processing},
 year = {2022}
}
```