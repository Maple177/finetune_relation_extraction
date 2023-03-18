import logging
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import (BertPreTrainedModel, BertModel)
from utils_data import class_weights

logger = logging.getLogger(__name__)

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,config,task_name,num_labels=14):
        super().__init__(config)
        self.num_labels = num_labels

        # use pure BERT; 
       
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)

        assert task_name in class_weights, "Unknown corpus: Add class weights manually to 'class_weights' defined in bert_model.py."
        assert len(class_weights[task_name]) == num_labels, "the number of labels NOT equal to the number of class weights."
        self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(np.log(class_weights[task_name])))     

        self.init_weights()
        #logger.info("BERT model initialised.")


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1,self.num_labels))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

