from argparse import ArgumentParser

def get_args():
     parser = ArgumentParser(description='Finetuning BERT models')
	
     group = parser.add_argument_group('--fine_tuning_options')
     group.add_argument("--data_dir", default="./data/", type=str,
                         help="The input data directory. Should contain tsv files. See instructions.")
     group.add_argument("--task_name", default=None, type=str,
		         help="Name of the corpus to use.")
     group.add_argument("--pretrained_model_path",default="./pretrained_models/",type=str,
                         help="Path to local pre-trained model weights.")
     group.add_argument("--model_name",type=str,help="Name of the BERT variant to use.")
     group.add_argument("--output_dir", default="./models/", type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
     group.add_argument("--dry_run",action="store_true",
                         help="Set this for a quick complete pass of codes; to quickly examine if bugs exist.")
     group.add_argument("--number_of_examples_for_dry_run",type=int,default=50,
                         help="Choose how many examples for dry run.")
     group.add_argument("--monitor",type=str,default="score",
                         help="Criteria to use on the validation set to monitor early stopping.")
     group.add_argument("--early_stopping",action="store_true",
                         help="If to use early stopping during training.")
     group.add_argument("--patience",type=int,default=5,
                         help="Patience of early stopping")
     group.add_argument("--ensemble_size",type=int,
                         help="Number of models i.e. runs initialized with different random seeds.")
     group.add_argument("--max_seq_length",type=int,default=512,
                         help="Sequences longer than this value (number of wordpieces) will be truncated.")
     group.add_argument("--batch_size", default=16, type=int,
                         help="Batch size per GPU/CPU for training.")
     group.add_argument("--learning_rate",type=float,default=2e-5)
     group.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     group.add_argument("--num_train_epochs", default=20, type=int,
                         help="Total number of training epochs; Maximum training epochs in case of using early stopping.")
     group.add_argument("--seed",type=int,default=41,
		         help="The random seed to initialize model weights.")
     group.add_argument("--shuffle_train",action="store_true",help="If set, shuffle the train set before training.")
     group.add_argument("--warmup",action="store_true",help="If set, use linear warmup scheduler for learning rate.")
     group.add_argument("--warmup_ratio",type=float,default=0.1,
		         help="Denote this value by p; p * number_of_total_steps will be used for warmup i.e. "
		              "at the end of p * number_of_total_steps, learning rate will attain the target value.")
     group.add_argument("--logging_steps", type=int, default=50,
                         help="Denote this value by X; Log the training loss every X updates steps.")
     group.add_argument("--do_not_save_all_models",action="store_true",
                         help="Set this to save disk space: if not set, we save ensemble_size checkpoints for all runs "
                              "(which may require too much disk space).")
     args = parser.parse_args()
     return args
