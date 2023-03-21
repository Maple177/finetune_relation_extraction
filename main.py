from __future__ import absolute_import, division, print_function

import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig
from bert_model import BertForSequenceClassification
from sklearn.metrics import f1_score
from opt import get_args
from loader import DataLoader
from utils import (set_seed, load_data, train, evaluate, summarize_results)
from utils_data import bert_names_to_versions, number_of_labels

logger = logging.getLogger(__name__)

def main():
    args = get_args()
    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    args.num_labels = number_of_labels[args.task_name]

    if not os.path.exists("logging"):
        os.makedirs("logging")
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("scores"):
        os.makedirs("scores")

    # for a quick test, use less epochs and a smaller ensemble size
    if args.dry_run:
        args.ensemble_size = 2
        args.num_train_epochs = 2

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=f"logging/log_{args.task_name}_{args.model_name}",filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    # load pre-trained model weights if saved locally
    pretrained_model_path = os.path.join(args.pretrained_model_path,args.model_name)
    if os.path.exists(pretrained_model_path):
        config_name_or_path = os.path.join(pretrained_model_path,"config.json")
        model_name_or_path = pretrained_model_path
    else:
        config_name_or_path = bert_names_to_versions[args.model_name]
        model_name_or_path = bert_names_to_versions[args.model_name]
    
    config = BertConfig.from_pretrained(config_name_or_path,num_labels=args.num_labels)
    
    if args.warmup:
        hyperparameter_combination = f"{args.task_name}_{args.model_name}_{args.batch_size}_{args.learning_rate}_{args.seed}_with_warmup"
    else:
        hyperparameter_combination = f"{args.task_name}_{args.model_name}_{args.batch_size}_{args.learning_rate}_{args.seed}_no_warmup"
    output_dir = os.path.join(args.output_dir,hyperparameter_combination)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    train_dataloader, dev_dataloader, test_dataloader = load_data(args)
    
    dev_preds = []
    test_preds = []
    logger.info(f"start training...seed:{args.seed}")
    for nr in range(args.ensemble_size):
        torch.cuda.empty_cache()
        set_seed(args,nr)
        model = BertForSequenceClassification.from_pretrained(model_name_or_path,
                                                              config=config,
                                                              task_name=args.task_name,
                                                              num_labels=args.num_labels)
        model.to(args.device)
        logger.info(f"training model-{nr+1}.")

        checkpoint, _, _, _ = train(args,train_dataloader,dev_dataloader,model,nr+1)
        
    
        model = BertForSequenceClassification.from_pretrained(checkpoint,
                                                              config=config,
                                                              task_name=args.task_name,
                                                              num_labels=args.num_labels)
        model.to(args.device)
  
        dev_preds.append(np.expand_dims(evaluate(dev_dataloader,model,args.num_labels,eval=True),0)) 
        test_preds.append(np.expand_dims(evaluate(test_dataloader,model,args.num_labels,predict_only=True),0))
    

    with open(os.path.join(output_dir,"dev_preds.npy"), "wb") as fp:
        np.save(fp,np.concatenate(dev_preds,0))
    with open(os.path.join(output_dir,"test_preds.npy"),"wb") as fp:
        np.save(fp,np.concatenate(test_preds,0))
   
    # calculate evaluation scores on the prediction files.
    score_path = os.path.join("scores",hyperparameter_combination)
    if not os.path.exists(score_path):
        os.makedirs(score_path)
    # printout micro- scores
    logger.info(f"finished. evaluatoin scores are saved under {score_path}.")
    logger.info("following are evaluation scores (micro-): avg ± std.")
    if args.do_not_generate_test_score:
        dev_score_report = summarize_results(args,output_dir,score_path) 
        logger.info(dev_score_report)
    else:
        dev_score_report, test_score_report = summarize_results(args,output_dir,score_path) 
        logger.info(dev_score_report)
        logger.info(test_score_report)   
 
if __name__ == "__main__":
    main()
