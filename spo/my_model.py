import torch
import torch.nn as nn
from transformers import BertModel,BertPreTrainedModel


class SubjectModel(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dense = nn.Linear(768,2)

    def forward(self,input_ids,attention_mask):
        output = self.bert(input_ids,attention_mask=attention_mask)
        subject_out = self.dense(output[0])
        subject_out = torch.sigmoid(subject_out)
        return output[0],subject_out

class ObjectModel(nn.Module):
    def __init__(self,subject_model):
        super().__init__()
        self.encoder = subject_model
        self.subject_encoding = nn.Linear(2,768)
        self.dense_object = nn.Linear(768,49*2)

    def forward(self,input_ids,subject_position,attention_mask):
        output,subject_out = self.encoder(input_ids,attention_mask=attention_mask)
        subject_position = self.subject_encoding(subject_position).unsqueeze(1)
        object_out = output+subject_position
        object_out = self.dense_object(object_out)
        object_out = torch.reshape(object_out,(object_out[0],object_out[1],49,2))
        object_out = torch.sigmoid(object_out)
        object_out = torch.pow(object_out,4)

        return subject_out,object_out





