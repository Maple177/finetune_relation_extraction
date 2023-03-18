# Finetune_relation_extraction
codes for fine-tuning domain-specific BERT variants on relation extraction (RE) datasets.

# About Fine-tuning
- we take sequtial seeds for multiple runs, e.g. if seeded by 41 for 5 runs, the 5 runs will be respectively seeded by 41 ,42, 43, 44, 45.
- we calculate the weighted binary cross entropy as the loss with the weight of the $i$-th class: $c_i=\frac{\sum\limits_{j=0}^{M-1} N_j}{N_i}$ , where $N_i$ is the number of examples labelled by the $i$-th class; $M$ is the number of classes.
- we use a slanted triangular scheduler on learning rate (remove --warmup to change to constant learning rate).

# How to use
- create a directory /data/ under the current directory, prepare train.tsv, dev.tsv, test.tsv under /data/corpus_name/, e.g. /data/chemprot_blue/. 

  :exclamation: make sure that the first line of your tsv files contains column names seperated by '\t'.

- we keep a list of corpus and BERT variants as presets, you can input directly these corpus names and BERT variant names.

| | |
| ---| --- |
| corpus| chemprot_blurb \| chemprot_blue \| ddi_blurb \| ddi_blue \| i2b2 \| i2b2_modified  |
|  BERT variants | biobert \| scibert \| pubmedbert \| bluebert \| clinicalbert \| biolinkbert |

- if you want to load pre-trained models from local files, create a directory /pretrained_models/ and put vocal.txt, pytorch_model.bin and config.json under /pretrained_models/model_name/ i.e. /pretrained_models/biobert/.

- if you use datasets or BERT variants that are not in the pre-list, simple modifications need to be made to codes:
  - add a map from labels to integers in the dictionary "label2id" in utils_data.py. Notice that you should ALWAYS map the false relation to 0 (in our evaluation we assume that 0 refers to the false relation and is thus excluded). An example label-to-id map for chemprot (blurb / blue) is:
   ```
   {"false": 0, 
    "CPR:3": 1, 
    "CPR:4": 2, 
    "CPR:5": 3, 
    "CPR:6": 4, 
    "CPR:9": 5}
   ```
  - similarly, add the class weights in "class_weights";
  - for a new BERT variant, add its version name on HuggingFace in "bert_names_to_versions";
  - number of classes (including false relation) in "number_of_labels".
  
# Command
- use example slurm file to train an ensemble of 5 PubMedBERT models on chemprot (blurb). Set your hyperparameters in the slurm file or directory pass them using the following line.
```
python3 main.py --model_name pubmedbert \
                --task_name chemprot_blurb \
                --num_train_epochs 60 \
                --seed 41  \
                --learning_rate 1e-05 \
                --ensemble_size 5  --warmup \
                --do_not_save_all_models
```
- use the example command line should be able to reproduce the follwing results (test_scores.csv) on ChemProt (blurb):


| | precision | recall | F1-score |
| --- | --- | --- | --- |
| run-1 | 0.795 | 0.747 | 0.770 |
| run-2 | 0.790 | 0.764 | 0.777 |
| run-3 | 0.788 | 0.762 | 0.775 |
| run-4 | 0.784 | 0.758 | 0.771 |
| run-5 | 0.803 | 0.749 | 0.775 |
| vote | 0.814 | 0.759 | 0.785 |


- set --dry_run to make a quick pass (this will take a small subset of data), check:
  - if dev_preds.npy & test_preds.npy are correctly generated under /models/your_hyperparamater_setting/; 
  - if dev_scores.csv & test_scores.csv are correctly generated under /scores/your_hyperparameter_setting/.
  
- set --early_stopping to enable early stopping, by default fine-tuning will stop if micro F1-score on the validation set does not increase within 3 epochs.

- get more details about fine-tuning options:
```
python3 main.py --help
```

# Environment
- Python 3.8.5
- Pytorch 1.13.0
- Transformers 4.6.0

(you can create a conda environment using environment.yml)

# Output

- by default, best checkpoints of each run and predictions on dev / test will be saved under the directory /models/.
- you can monitor the process of fine-tuning by checking the log under /logging/.
- evaluation results on the validation set and the test set will be saved in csv files under /scores/.

# Evaluation

:exclamation: by default we calculate the micro F1-score with the false relation EXCLUDED.

if you do not need saved model weights and prediction files, to release disk space, you can then delete everything under /models/.
