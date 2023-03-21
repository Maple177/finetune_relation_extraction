from argparse import ArgumentParser


def get_args():
     parser = ArgumentParser(description='Finetuning BERT models')
	
     group = parser.add_argument_group('--train_options')
     group.add_argument("--data_dir", default="./data/", type=str,
                         help="The input data dir. Should contain the .tsv files for the task.")
     group.add_argument("--config_dir", default="./config/", type=str,
                         help="Path to pre-trained config or shortcut name selected in the list.")
     group.add_argument("--task_name", default=None, type=str)
     group.add_argument("--pretrained_model_path",default="./pretrained_models/",type=str,
                         help="Path to locally saved pre-trained model weights.")
     group.add_argument("--model_name",type=str,help="Name of the BERT variant to use.")
     group.add_argument("--output_dir", default="./models/", type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
     group.add_argument("--dry_run",action="store_true",
                         help="Set this for a quick complete pass of codes; to quickly examine if bugs exist.")
     group.add_argument("--number_of_examples_for_dry_run",type=int,default=50,
                         help="Choose a small subset of datasets for dry run.")
     group.add_argument("--monitor",type=str,default="score",
                         help="Criteria to use for early stopping.")
     group.add_argument("--early_stopping",action="store_true",
                         help="If use early stopping during training.")
     group.add_argument("--patience",type=int,default=5,
                         help="Patience of early stopping")
     group.add_argument("--ensemble_size",type=int,
                         help="Number of models i.e. runs initialized with different random seeds.")
     group.add_argument("--max_seq_length",type=int,default=512,
                         help="Sequences longer than this value will be truncated.")
     group.add_argument("--batch_size", default=16, type=int,
                         help="Batch size per GPU/CPU for training.")
     group.add_argument("--learning_rate",type=float,default=2e-5)
     group.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     group.add_argument("--max_num_epochs",default=15,type=int,
                         help="Maximum number of epochs.")
     group.add_argument("--num_train_epochs", default=20, type=int,
                         help="Total number of training epochs to perform.")
     group.add_argument("--seed",type=int)
     group.add_argument("--shuffle_train",action="store_true",help="If set, shuffle the train set before training.")
     group.add_argument("--warmup",action="store_true",help="If set, use linear warmup scheduler for learning rate.")
     group.add_argument("--warmup_ratio",type=float,default=0.1)
     group.add_argument("--logging_steps", type=int, default=50,
                         help="Log every X updates steps.")
     group.add_argument("--do_not_save_all_models",action="store_true",
                         help="Set this to save storage space: if not set, we save ensemble_size checkpoints for each run"
                              "may take much storage space.")
     args = parser.parse_args()
     return args
