import logging
import os
import random
import numpy as np
import pandas as pd
import json
import torch
from torch.optim import AdamW
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score,precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from loader import DataLoader
from utils_data import (bert_names_to_versions, label2id)

logger = logging.getLogger(__name__)

def one_hot(x,num_labels):
    res = np.zeros((len(x),num_labels),dtype=np.int)
    for i, v in enumerate(x):
        res[i,v] = 1
    return res

def get_scores(gold,pred,num_labels):
    scores_micro_plus = precision_recall_fscore_support(gold,pred,average="micro")[:3]
    scores_micro_minus = precision_recall_fscore_support(gold,pred,average="micro",labels=list(range(1,num_labels)))[:3]
    scores_macro_plus = precision_recall_fscore_support(gold,pred,average="macro")[:3]
    scores_macro_minus = precision_recall_fscore_support(gold,pred,average="macro",labels=list(range(1,num_labels)))[:3]
    return scores_micro_plus, scores_micro_minus, scores_macro_plus, scores_macro_minus

def generate_dataframe(ps,rs,fs,ensemble_size,output_path):
    df_score = pd.DataFrame({"runs":[f"run-{i}" for i in range(1,ensemble_size+1)]+["vote"],"precision":ps,"recall":rs,"F1-score":fs})
    df_score.to_csv(output_path,index=False)

def summarize_results(args,path,output_path):
    num_labels = args.num_labels
    mlb = MultiLabelBinarizer(classes=list(range(num_labels)))
    
    dev_output_path = os.path.join(output_path,"dev")
    if not os.path.exists(dev_output_path):
        os.makedirs(dev_output_path)
    y_dev = mlb.fit_transform(json.load(open(os.path.join(args.data_dir,args.task_name,"label","dev.json"),'r'))["labels"])
    if args.dry_run:
        y_dev = y_dev[:args.number_of_examples_for_dry_run]
    dev_preds = np.load(os.path.join(path,"dev_preds.npy"))
    ensemble_size, dev_size, _ = dev_preds.shape
    
    if not args.do_not_generate_test_score:
        test_output_path = os.path.join(output_path,"test")
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)
        y_test = mlb.fit_transform(json.load(open(os.path.join(args.data_dir,args.task_name,"label","test.json"),'r'))["labels"])
        if args.dry_run:
            y_test = y_test[:args.number_of_examples_for_dry_run]    
        test_preds = np.load(os.path.join(path,"test_preds.npy"))
        _, test_size = test_preds.shape

    assert ensemble_size == args.ensemble_size, f"should have {args.ensemble_size} runs; only got {ensemble_size} runs." 
    
    # evaluation on dev
    dev_vote_preds = one_hot(np.argmax(dev_preds.sum(0),1),num_labels)
    dev_ps_micro_plus, dev_rs_micro_plus, dev_fs_micro_plus = [], [], []
    dev_ps_micro_minus, dev_rs_micro_minus, dev_fs_micro_minus = [], [], []
    dev_ps_macro_plus, dev_rs_macro_plus, dev_fs_macro_plus = [], [], []
    dev_ps_macro_minus, dev_rs_macro_minus, dev_fs_macro_minus = [], [], []
    
    for pred in [dev_preds[i] for i in range(ensemble_size)] + [dev_vote_preds]:
        scores_micro_plus, scores_micro_minus, scores_macro_plus, scores_macro_minus = get_scores(y_dev,pred,num_labels)
        p, r, f = scores_micro_plus[:3]
        dev_ps_micro_plus.append(p); dev_rs_micro_plus.append(r); dev_fs_micro_plus.append(f)
        p, r, f = scores_micro_minus[:3]
        dev_ps_micro_minus.append(p); dev_rs_micro_minus.append(r); dev_fs_micro_minus.append(f)
        p, r, f = scores_macro_plus[:3]
        dev_ps_macro_plus.append(p); dev_rs_macro_plus.append(r); dev_fs_macro_plus.append(f)
        p, r, f = scores_macro_minus[:3]
        dev_ps_macro_minus.append(p); dev_rs_macro_minus.append(r); dev_fs_macro_minus.append(f)
    
    generate_dataframe(dev_ps_micro_plus,dev_rs_micro_plus,dev_fs_micro_plus,ensemble_size,os.path.join(dev_output_path,"micro_plus.csv"))
    generate_dataframe(dev_ps_micro_minus,dev_rs_micro_minus,dev_fs_micro_minus,ensemble_size,os.path.join(dev_output_path,"micro_minus.csv"))
    generate_dataframe(dev_ps_macro_plus,dev_rs_macro_plus,dev_fs_macro_plus,ensemble_size,os.path.join(dev_output_path,"macro_plus.csv"))
    generate_dataframe(dev_ps_macro_minus,dev_rs_macro_minus,dev_fs_macro_minus,ensemble_size,os.path.join(dev_output_path,"macro_minus.csv"))
    
    dev_f1_scores = pd.read_csv(os.path.join(dev_output_path,"micro_minus.csv"))["F1-score"].values
    dev_run_scores, dev_vote_score = dev_f1_scores[:ensemble_size], dev_f1_scores[-1]
    dev_score_report = f"on dev: {np.mean(dev_run_scores)} ± {np.std(dev_run_scores)}; voting score: {dev_vote_score}"
    
    if not args.do_not_generate_test_score:
        # evaluation on test
        test_vote_preds = np.zeros((test_size,num_labels))
        all_test_preds = []
        for i in range(ensemble_size):
            tmp_test_pred = one_hot(test_preds[i],num_labels)
            test_vote_preds += tmp_test_pred
            all_test_preds.append(tmp_test_pred)
        test_vote_preds = one_hot(np.argmax(test_vote_preds,1),num_labels)
        all_test_preds.append(test_vote_preds)

        test_ps_micro_plus, test_rs_micro_plus, test_fs_micro_plus = [], [], []
        test_ps_micro_minus, test_rs_micro_minus, test_fs_micro_minus = [], [], []
        test_ps_macro_plus, test_rs_macro_plus, test_fs_macro_plus = [], [], []
        test_ps_macro_minus, test_rs_macro_minus, test_fs_macro_minus = [], [], []

        for pred in all_test_preds:
            scores_micro_plus, scores_micro_minus, scores_macro_plus, scores_macro_minus = get_scores(y_test,pred,num_labels)
            p, r, f = scores_micro_plus[:3]
            test_ps_micro_plus.append(p); test_rs_micro_plus.append(r); test_fs_micro_plus.append(f)
            p, r, f = scores_micro_minus[:3]
            test_ps_micro_minus.append(p); test_rs_micro_minus.append(r); test_fs_micro_minus.append(f)
            p, r, f = scores_macro_plus[:3]
            test_ps_macro_plus.append(p); test_rs_macro_plus.append(r); test_fs_macro_plus.append(f)
            p, r, f = scores_macro_minus[:3]
            test_ps_macro_minus.append(p); test_rs_macro_minus.append(r); test_fs_macro_minus.append(f)

        generate_dataframe(test_ps_micro_plus,test_rs_micro_plus,test_fs_micro_plus,ensemble_size,os.path.join(test_output_path,"micro_plus.csv"))
        generate_dataframe(test_ps_micro_minus,test_rs_micro_minus,test_fs_micro_minus,ensemble_size,os.path.join(test_output_path,"micro_minus.csv"))
        generate_dataframe(test_ps_macro_plus,test_rs_macro_plus,test_fs_macro_plus,ensemble_size,os.path.join(test_output_path,"macro_plus.csv"))
        generate_dataframe(test_ps_macro_minus,test_rs_macro_minus,test_fs_macro_minus,ensemble_size,os.path.join(test_output_path,"macro_minus.csv"))

        test_f1_scores = pd.read_csv(os.path.join(test_output_path,"micro_minus.csv"))["F1-score"].values
        test_run_scores, test_vote_score = test_f1_scores[:ensemble_size], test_f1_scores[-1]
        test_score_report = f"on test: {np.mean(test_run_scores)} ± {np.std(test_run_scores)}; voting score: {test_vote_score}"
        return dev_score_report, test_score_report
    return dev_score_report

def set_seed(args,ensemble_id):
    seed = args.seed + ensemble_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def get_wp_ids(sentences,tokenizer,cls_id,sep_id):
    wp_ids = []
    for sentence in sentences:
        wp_id = [cls_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)) + [sep_id]
        wp_ids.append(wp_id)
    return wp_ids

def extract_labels(df,label2id):
    return [[label2id[l]] for l in df.label.values]

def load_data(args):
    data_dir = os.path.join(args.data_dir,args.task_name)
    assert os.path.exists(data_dir), "unknown corpus. To test on this corpus, add requested modifications to utils_data.py (see instructions on the GitHub page)."
    if not os.path.exists(os.path.join(data_dir,"cache",args.model_name)):
        logger.info("first loading: reading tsv files...")
        # read tsv files
        df_train = pd.read_csv(os.path.join(data_dir,"train.tsv"),sep='\t')
        df_dev = pd.read_csv(os.path.join(data_dir,"dev.tsv"),sep='\t')
        df_test = pd.read_csv(os.path.join(data_dir,"test.tsv"),sep='\t')
        # save labels
        l2d = label2id[args.task_name]
        y_train = extract_labels(df_train,l2d)
        y_dev = extract_labels(df_dev,l2d)
        y_test = extract_labels(df_test,l2d)
        label_dir = os.path.join(data_dir,"label")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        json.dump({"labels":y_train},open(os.path.join(label_dir,"train.json"),'w'))
        json.dump({"labels":y_dev},open(os.path.join(label_dir,"dev.json"),'w'))
        json.dump({"labels":y_test},open(os.path.join(label_dir,"test.json"),'w'))
        # caching
        if not os.path.exists(os.path.join(data_dir,"cache")):
            os.makedirs(os.path.join(data_dir,"cache"))
        logger.info(f"{args.model_name}:tokenization...")
        # load from locally saved files or download using internet
        tokenizer_path = os.path.join(args.pretrained_model_path,args.model_name)
        if os.path.exists(tokenizer_path):
            tokenizer_name_or_path = tokenizer_path
        else:
            tokenizer_name_or_path = bert_names_to_versions[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        # tokenization 
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        train_wp_ids = get_wp_ids(df_train.sentence.values,tokenizer,cls_id,sep_id)
        dev_wp_ids = get_wp_ids(df_dev.sentence.values,tokenizer,cls_id,sep_id)
        test_wp_ids = get_wp_ids(df_test.sentence.values,tokenizer,cls_id,sep_id)
        dst_dir = os.path.join(data_dir,"cache",args.model_name)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        json.dump({"wp_ids":train_wp_ids},open(os.path.join(dst_dir,"train.json"),'w'))
        json.dump({"wp_ids":dev_wp_ids},open(os.path.join(dst_dir,"dev.json"),'w'))
        json.dump({"wp_ids":test_wp_ids},open(os.path.join(dst_dir,"test.json"),'w'))
    else:
        logger.info("loaded from cache.")
    train_dataloader = DataLoader(args,"train")
    dev_dataloader = DataLoader(args,"dev",eval=True)
    test_dataloader = DataLoader(args,"test",inference=True)
    return train_dataloader, dev_dataloader, test_dataloader

def evaluate(dataloader,model,num_labels,eval=False,predict_only=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    full_preds = []
    full_golds = [] # gold standard
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            if predict_only:
                logits = model(**batch)[0]
            else:
                loss, logits = model(**batch)[:2]
            preds = logits.detach().cpu().numpy()
            if not predict_only:
               eval_loss += loss.item()
               nb_eval_steps += 1
               preds = one_hot(np.argmax(preds,axis=1),num_labels)
               full_golds.append(batch["labels"].detach().cpu().numpy())
            else:
               preds = np.argmax(preds,axis=1)
            full_preds.append(preds)
    full_preds = np.concatenate(full_preds)
    if not predict_only and not eval:
        # during training
        eval_loss = eval_loss / nb_eval_steps
        full_golds = np.concatenate(full_golds)
    if predict_only or eval:
        return full_preds
    else:
        return eval_loss, f1_score(full_golds,full_preds,average="micro",labels=list(range(1,num_labels)))
        
def train(args,train_dataloader,dev_dataloader,model,ensemble_id):
    NUM_EPOCHS = args.num_train_epochs

    n_params = sum([p.nelement() for p in model.parameters()])
    logger.info(f'* number of parameters: {n_params}')

    optimizer = AdamW(model.parameters(),lr=args.learning_rate)
    logger.info(f"learning rate: {args.learning_rate}")

    if args.warmup:
        t_total = len(train_dataloader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_ratio*t_total,num_training_steps=t_total)
        logger.info(f"number of epochs:{NUM_EPOCHS}; number of steps:{t_total}")
  
    if args.early_stopping:
        logger.info("use early stopping.")

    if args.do_not_save_all_models:
        if args.warmup:
            best_model_dir = f"{args.task_name}_{args.model_name}_{args.batch_size}_{args.learning_rate}_{args.seed}_with_warmup/best_ckpt"
        else:
            best_model_dir = f"{args.task_name}_{args.model_name}_{args.batch_size}_{args.learning_rate}_{args.seed}_no_warmup/best_ckpt"
    else: 
        if args.warmup: 
            best_model_dir = f"{args.task_name}_{args.model_name}_{args.batch_size}_{args.learning_rate}_{args.seed}_with_warmup/run_{ensemble_id}"
        else:
            best_model_dir = f"{args.task_name}_{args.model_name}_{args.batch_size}_{args.learning_rate}_{args.seed}_no_warmup/run_{ensemble_id}"
    
    output_dir = os.path.join(args.output_dir,best_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", NUM_EPOCHS)
    logger.info("  Batch size = %d", args.batch_size)
    #if args.gradient_accumulation:
    #    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    logging_loss, min_loss, prev_dev_loss = 0.0, np.inf, np.inf
    max_score, prev_dev_score = -np.inf, -np.inf
    training_hist = []
    model.zero_grad()

    dev_loss_record = []
    dev_score_record = []

    for epoch in range(NUM_EPOCHS):
        tr_loss = 0.0
        logging_loss = 0.0
        grad_norm = 0.0
        for step, batch in enumerate(train_dataloader): 
            model.train()
            loss = model(**batch)[0] 

            loss.backward() # gradient will be stored in the network
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)

            grad_norm += gnorm
                                                
            tr_loss += loss.item()

            optimizer.step()
            if args.warmup:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                # Log metrics
                logger.info(f"training loss = {(tr_loss - logging_loss)/args.logging_steps} | global step = {global_step}")
                logging_loss = tr_loss

        dev_loss, dev_score = evaluate(dev_dataloader,model,args.num_labels)
        dev_loss_record.append(dev_loss)
        dev_score_record.append(dev_score)

        if args.warmup:
            logger.info(f"current lr = {scheduler.get_lr()[0]}")
        logger.info(f"validation loss = {dev_loss} | validation F1-score = {dev_score} | ensemble_id = {ensemble_id} epoch = {epoch}")
        
        if args.monitor == "loss" and dev_loss < min_loss:
            min_loss = dev_loss
            best_epoch = epoch
            
            # save model
            model.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir,'training_args.bin'))
            logger.info("new best model! saved.")
        
        if args.monitor == "score" and dev_score > max_score:
            max_score = dev_score
            best_epoch = epoch

            # save model
            model.save_pretrained(output_dir)
            torch.save(args,os.path.join(output_dir,"training_args.bin"))
            logger.info("new best model! saved.")
        
        if args.early_stopping and args.monitor == "loss":
            if dev_loss < prev_dev_loss:
                training_hist.append(True)
            else:
                training_hist.append(False)
                if len(training_hist) > args.patience and not np.any(training_hist[-args.patience:]):
                    logger.info(f"early stopping triggered: best loss on validation set: {min_loss} at epoch {best_epoch}.")
                    #train_iterator.close()
                    break
            prev_dev_loss = dev_loss

        if args.early_stopping and args.monitor == "score":
            if dev_score >= prev_dev_score:
                training_hist.append(True)
            else:
                training_hist.append(False)
                if len(training_hist) > args.patience and not np.any(training_hist[-args.patience:]):
                    logger.info(f"early stopping triggered: best F-score on validation set: {max_score} at {best_epoch}.")
                    #train_iterator.close()
                    break
            prev_dev_score = dev_score

        if epoch + 1 == NUM_EPOCHS:
            break

    return output_dir, dev_loss_record, dev_score_record, best_epoch
