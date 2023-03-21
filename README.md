# Finetune_relation_extraction
codes for fine-tuning domain-specific BERT variants on relation extraction (RE) datasets.

(applies also to text classification datasets)

# About Fine-tuning
- we use consecutive seeds for multiple runs, e.g. if seeded by 41 for 5 runs, the 5 runs will be respectively seeded by 41 ,42, 43, 44, 45.
- we use a weighted binary cross entropy as the loss with the weight of the $i$-th class: $c_i=\frac{\sum\limits_{j=0}^{M-1} N_j}{N_i}$ , where $N_i$ is the number of examples labelled by the $i$-th class; $M$ is the number of classes.
- we use a slanted triangular scheduler on the learning rate (remove --warmup to change to constant learning rate).

# How to use
- create a directory /data/ under the current directory, prepare train.tsv, dev.tsv, test.tsv under /data/corpus_name/, e.g. /data/chemprot_blue/. 

  :exclamation: make sure that:
  - the first line of your tsv files contains column names seperated by '\t'; 
  - there must be two columns named by "sentence" and "label". (refer to example.tsv as an example; if labels are not available, just fill the "label" column by random existing labels)

- we keep a list of corpus and BERT variants as presets, you can input directly these corpus names and BERT variant names.

| | |
| ---| --- |
| corpus| chemprot_blurb \| chemprot_blue \| ddi_blurb \| ddi_blue \| i2b2 \| i2b2_modified  |
|  BERT variants | biobert \| scibert \| pubmedbert \| bluebert \| clinicalbert \| biolinkbert |

- :raised_hand: if you want to load pre-trained models from local files (or if you have no internet connection), create a directory /pretrained_models/ and put vocal.txt, pytorch_model.bin and config.json under /pretrained_models/model_name/ e.g. /pretrained_models/biobert/.

- :raised_hand: if you use datasets or BERT variants that are not in the pre-list, simple modifications need to be made:
  - add a map from labels to integer ids in the dictionary "label2id" in utils_data.py. Notice that you should ALWAYS map the false relation to 0 (in our evaluation we assume that 0 refers to the false relation and is thus excluded in some cases). An example label-to-id map for chemprot (blurb / blue) is:
   ```
   {"false": 0, 
    "CPR:3": 1, 
    "CPR:4": 2, 
    "CPR:5": 3, 
    "CPR:6": 4, 
    "CPR:9": 5}
   ```
  - similarly, add the class weights in "class_weights";
  - for a new BERT variant, add its Huggingface version name in "bert_names_to_versions";
  - number of classes (including the false relation) in "number_of_labels".
  
# Command
- use the example slurm file will train an ensemble of 5 PubMedBERT models on ChemProt (blurb). Set your hyperparameters in the slurm file or directory pass them using the following line.
```
python3 main.py --model_name pubmedbert \
                --task_name chemprot_blurb \
                --num_train_epochs 60 \
                --learning_rate 1e-05 \
                --ensemble_size 5  --warmup \
                --do_not_save_all_models
```
- use the example command line should be able to reproduce the follwing result (scores/.../test/micro_minus.csv) on ChemProt (blurb):


| | precision | recall | F1-score |
| --- | --- | --- | --- |
| run-1 | 0.795 | 0.747 | 0.770 |
| run-2 | 0.790 | 0.764 | 0.777 |
| run-3 | 0.788 | 0.762 | 0.775 |
| run-4 | 0.784 | 0.758 | 0.771 |
| run-5 | 0.803 | 0.749 | 0.775 |
| vote | 0.814 | 0.759 | 0.785 |


- :bulb: set --dry_run to make a quick pass (this will take a small subset of data), check:
  - if dev_preds.npy & test_preds.npy are correctly generated under /models/.../; 
  - if macro_minus.csv, macro_plus.csv, micro_minus.csv, micro_plus.csv are correctly generated under /scores/.../dev/ & /scores/.../test/.
  
- set --early_stopping to enable early stopping, by default fine-tuning will stop if micro F1-score on the validation set does not increase within 5 epochs.

- :bulb: by default best checkpoints of all runs will be saved under /models/.../run_i ($i$=1,...,ensemble_size), but this may take too much disk space. Set --do_not_save_all_models to save only the best checkpoint of the current run (if you do not need fine-tuned model weights for each run).

- get more details about fine-tuning options:
```
python3 main.py --help
```

# Environment
- Python 3.8.5
- Numpy 1.19.2
- Pandas 1.1.3
- Scikit-learn 0.23.2
- Pytorch 1.13.0
- Transformers 4.6.0

(you can create a conda environment using environment.yml)

# Output

- prediction files (label ids) on dev & test set will be saved under the directory /models/.
- you can monitor the process of fine-tuning by viewing the log under /logging/.
- evaluation results on the validation set and the test set will be saved respectively under /scores/.../dev/ and /scores/.../test/.

if you do not need saved model weights and prediction files, to release disk space, you can then delete everything under /models/.

# Evaluation

:paperclip: we calculate four types of F1-scores and save them automatically in .csv files after fine-tuning.

| name | definition |
| --- | --- |
| micro+ | micro F1-score |
| micro- | micro F1-score with the false relation EXCLUDED |
| macro+ | macro F1-score |
| macro- | macro F1-score with the false relation EXCLUDED |

- if you need other types of scores, you can calculate whatever scores you want using prediction files.
- :raised_hand: if labels of the test set are not available, set --do_not_generate_test_score to skip evaluation on the test set.
