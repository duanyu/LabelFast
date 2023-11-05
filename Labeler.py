import torch
import numpy as np
from modelscope.pipelines import pipeline
from tqdm import tqdm
from sklearn.metrics import accuracy_score  

class Labeler():
    def __init__(self):
        self.model = pipeline("text2text-generation", "damo/nlp_mt5_zero-shot-augment_chinese-base", model_revision="v1.0.0")
        print(f'mT5标注员已就位！')
        
    def get_prompt(self, text, schema):
        # text -> prompt
        prompt = f'''文本分类。\n候选标签：{','.join(schema)}\n文本内容：{text}'''
        return prompt
    
    def get_prediction(self, prompt):
        # prompt -> LLM -> prediction（text + ids) + confidence
        inputs = self.model.preprocess(prompt)
        out = self.model.forward(inputs)
        out_ids = out.sequences
        out_text = self.model.decode(out_ids.numpy()[0])

        prediction = {'input': inputs, 'output': {'text': out_text, 'ids': out_ids}}
        return prediction
        
    def get_confidence(self, prediction):
        # prompt + prediction -> calibrater LLM -> confidence of prediction（0-1, float)
        inputs = prediction['input']
        output_ids = prediction['output']['ids']
        
        logits = self.model.model.forward(inputs['input_ids'], inputs['attention_mask'], output_ids).logits.detach()
        probs = torch.softmax(logits, dim=2)
        logprobs = torch.log(probs)

        # Note：output ids开头两个固定为0, 259；decoder的预测要往左shift一位
        pred_logprob_first_token = logprobs[0][1][output_ids[0][2]]
        answer_logprob = torch.exp(pred_logprob_first_token)

        return float(answer_logprob.numpy())
    
    def get_threshold_table(self, res):
        threshold_table = []
        
        confidence_list = [d['Labeler_confidence'] for d in res]
        threshold_min, threshold_max = min(confidence_list), max(confidence_list)
        
        step = (threshold_max - threshold_min) / (10 + 1)  
        intervals = [threshold_min + i * step for i in range(10 + 1)]  
        
        for threshold in intervals:
            new_res = [d for d in res if d['Labeler_confidence'] >= threshold]
            coverage = len(new_res) / len(res)
            truth, pred = [d['label'] for d in new_res], [d['Labeler_prediction'] for d in new_res]
            acc = accuracy_score(truth, pred)
            threshold_table.append({'confidence threshold': threshold, 'acc': acc, 'coverage': coverage})
            
        return threshold_table
        
    def run(self, data, task_type, schema, truth_label = None):
        # print(f'标注任务导入完成：\nlen(data):{len(data)}, task_type: {task_type}, shcema: {schema}')
                 
        res = []
        for i in tqdm(range(len(data))):
            input_text = data[i]
            prompt = self.get_prompt(input_text, schema)
            prediction = self.get_prediction(prompt)
            confidence = self.get_confidence(prediction)
            
            # print(f'{i+1}th:')
            # print(f'input_text: {input_text}')
            # print(f'''Labeler prediction: {prediction['output']['text']}''')
            # print(f'confidence: {confidence}')
            # print()
            
            if truth_label is None:
                res.append({'input_text': input_text, 'Labeler_prediction': prediction['output']['text'], 'Labeler_confidence': confidence})
            else:
                res.append({'input_text': input_text, 'label': truth_label[i], 'Labeler_prediction': prediction['output']['text'], 'Labeler_confidence': confidence, 'is_labeler_right': 'Right' if truth_label[i] == prediction['output']['text'] else 'Wrong'})
               
        if truth_label is None:
            return res, []
        else:     
            return res, self.get_threshold_table(res)