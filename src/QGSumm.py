"""
self-supervised QGSumm model
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class QGSumm(nn.Module):
    def __init__(self, base_model, uq_model, add_patient_info, add_previous_notes):
        super(QGSumm,self).__init__()
        # tips: parameterize in run_QGSumm.py
        self.base_model = base_model # model(BartForConditionalGeneration_QGSumm) defined in Base-model.py
        self.uq_model = uq_model # a classification model to get prediction based on summary and the original note, respectively
        self.add_patient_info = add_patient_info # a boolean value to indicate whether to add patient information to the input of the base model
        self.add_previous_notes = add_previous_notes

    def forward(self, model_inputs, bart_tokenizer, lambda1):
        # model_inputs: data prepared in run_QGSumm.py -> (id, input_ids, attention_mask, patient_input_ids, patient_attention_mask, temporal_info_emb)
        # summary: summary output from the base model

        outputs = self.base_model.generate(*model_inputs, return_dict_in_generate=True, add_patient_info=self.add_patient_info, add_previous_notes=self.add_previous_notes)
        summary_ids, summary_onehot = outputs[0], outputs[1]

        summary = bart_tokenizer.decode(summary_ids, skip_special_tokens=True)

        summary_attention_mask = (summary_ids != 1).float()

        summary_emd = torch.matmul(summary_onehot, self.uq_model.longformer.embeddings.word_embeddings.weight)

        if summary_emd.shape[1] >= summary_attention_mask.shape[1]:
            raise ValueError("The length of sum_emb should be less than the length of sum_atten_mask")

        summary_attention_mask = summary_attention_mask[:, summary_ids.shape[1]-summary_emd.shape[1]:]

        global_attention_mask = torch.zeros(summary_emd.shape[:2])
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            prediction = self.uq_model(inputs_embeds = summary_emd, global_attention_mask = global_attention_mask, attention_mask = summary_attention_mask)
            target = self.uq_model(model_inputs['input_ids'], model_inputs['attention_mask'])
        # softmax= nn.Softmax()
        # prediction = softmax(prediction)
        # target = softmax(target)
        kl_loss_fct = nn.CrossEntropyLoss()
        kl_loss = kl_loss_fct(prediction, target)
        loss = torch.mean(kl_loss * (1+torch.exp((torch.sum(summary_attention_mask) - summary_attention_mask.shape[0])/(torch.sum(model_inputs['attention_mask']) - model_inputs['attention_mask'].shape[0]) - 0.5) * lambda1))
        return loss, summary


