import torch
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class Labeler_mt5():
    def __init__(self):
        # mt5标注员，支持文本分类任务；用cpu推理即可
        self.pipeline = pipeline("text2text-generation", "damo/nlp_mt5_zero-shot-augment_chinese-base", model_revision="v1.0.0")
        print('mt5标注员已就位！')

    def get_prompt(self, text, task_type, schema):
        # text -> prompt
        prompt = f'''文本分类。\n候选标签：{','.join(schema)}\n文本内容：{text}'''
        return prompt

    def get_prediction(self, prompt):
        # prompt -> LLM -> prediction（text + ids) + confidence
        inputs = self.pipeline.preprocess(prompt)
        out = self.pipeline.forward(inputs)
        out_ids = out.sequences
        out_text = self.pipeline.decode(out_ids.numpy()[0])

        prediction = {'input': inputs, 'output': {'text': out_text, 'ids': out_ids}}
        return prediction

    def get_first_token_confidence(self, prediction):
        # prompt + prediction -> calibrater LLM -> confidence of prediction（0-1, float)
        inputs = prediction['input']
        output_ids = prediction['output']['ids']

        logits = self.pipeline.model.forward(inputs['input_ids'], inputs['attention_mask'], output_ids).logits.detach()
        probs = torch.softmax(logits, dim=2)
        logprobs = torch.log(probs)

        # Note：output ids开头两个固定为0, 259；decoder的预测要往左shift一位
        pred_logprob_first_token = logprobs[0][1][output_ids[0][2]]
        answer_logprob = torch.exp(pred_logprob_first_token)

        return float(answer_logprob.numpy())

class Labeler_seqgpt():
    def __init__(self):
        # seg-gpt标注员，支持文本分类、NER任务；必须采取gpu推理，否则无法运行
        if not torch.cuda.is_available():
            raise RuntimeError('无CUDA，而seq-gpt必须进行GPU推理！')
        self.pipeline = pipeline(Tasks.text_generation, "damo/nlp_seqgpt-560m", model_revision="v1.0.1", run_kwargs={'gen_token': '[GEN]'})
        self.pipeline.model.to("cuda")
        print('seq-gpt标注员已就位！')

    def get_prompt(self, text, task_type, schema):
        # text -> prompt
        # task可选值为 抽取、分类。text为需要分析的文本。labels为类型列表，中文逗号分隔。
        if task_type == 'CLS':
            inputs = {'task': '分类', 'text': text, 'labels': '，'.join(schema)}
        elif task_type == 'NER':
            inputs = {'task': '抽取', 'text': text, 'labels': '，'.join(schema)}
        PROMPT_TEMPLATE = '输入: {text}\n{task}: {labels}\n输出: '
        prompt = PROMPT_TEMPLATE.format(**inputs)
        return prompt

    def get_prediction(self, prompt):
        # prompt -> LLM -> prediction（text + ids) + confidence
        input_ids = self.pipeline.tokenizer(
            prompt + '[GEN]',
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024)
        input_ids.to('cuda')
        full_input = input_ids
        input_ids = input_ids.input_ids
        outputs = self.pipeline.model.generate(
            input_ids, num_beams=4, do_sample=False, max_new_tokens=256)

        decoded_sentences = self.pipeline.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        decoded_sentence = decoded_sentences[0]
        decoded_sentence = decoded_sentence[len(prompt):]

        prediction = {'input': full_input, 'output': {'text': decoded_sentence.strip(), 'ids': outputs[0][len(input_ids[0]):]}}
        return prediction

    def get_first_token_confidence(self, prediction):
        # 得到输出内容的第一个token的cofidence，分类任务可直接用
        # prompt + prediction -> calibrater LLM -> confidence of prediction（0-1, float)
        inputs = prediction['input']
        output_ids = prediction['output']['ids']
        logits = self.pipeline.model.forward(inputs['input_ids']).logits.detach()
        probs = torch.softmax(logits, dim=2)
        logprobs = torch.log(probs)

        # 取decoder最后一位的logprob，即输出标签的第一个token
        pred_logprob_first_token = logprobs[0][-1][output_ids[0]]
        answer_logprob = torch.exp(pred_logprob_first_token)

        return float(answer_logprob)

    def get_ner_confidence(self, prompt, prediction):
        pred_text = prediction['output']['text']
        ner_out = {}
        for res in pred_text.strip().split('\n'):
            entity_type, entities = res.split(':')[0], res.split(':')[1].strip().split('\t')
            entities = [e for e in entities if e != 'None']
            if len(entities) > 0:
                ner_out[entity_type] = []
                entity_start_index = pred_text.find(entity_type+':') + len(entity_type+':')
                for e_i, entity in enumerate(entities):
                    entity_input_text = prompt + pred_text[:entity_start_index]
                    if e_i == 0:
                        # 第一个实体会用空格隔开
                        entity_input_output_text = entity_input_text + ' ' + entity
                    else:
                        entity_input_output_text = entity_input_text + entity

                    entity_input_outputs = self.pipeline.tokenizer(entity_input_output_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
                    entity_input_output_ids = entity_input_outputs['input_ids'][0].numpy()

                    # 对第一个实体，用冒号处理，其他的实体，用\t-188
                    if e_i == 0:
                        colon_index = np.where(entity_input_output_ids == 29)[0][-1]
                    else:
                        colon_index = np.where(entity_input_output_ids == 188)[0][-1]

                    if entity_input_output_ids[colon_index + 1] == 210:
                        # 遇空格，则右移一位
                        entity_input_ids = torch.tensor(np.array([entity_input_output_ids[:colon_index+2]]))
                        entity_input_ids = entity_input_ids.to('cuda')
                        entity_output_ids = entity_input_output_ids[colon_index+2:]
                    else:
                        entity_input_ids = torch.tensor(np.array([entity_input_output_ids[:colon_index+1]]))
                        entity_input_ids = entity_input_ids.to('cuda')
                        entity_output_ids = entity_input_output_ids[colon_index+1:]

                    entity_pred = {'input': {'input_ids': entity_input_ids}, 'output': {'ids': entity_output_ids}}
                    first_token_confidence = self.get_first_token_confidence(entity_pred)

                    ner_out[entity_type].append([entity, first_token_confidence])

                    if e_i == 0:
                        entity_start_index += len(' '+entity+'\t')
                    else:
                        entity_start_index += len(entity+'\t')

        return ner_out


class Labeler():
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'mt5':
            self.labeler = Labeler_mt5()
        elif model_name == 'seq-gpt':
            self.labeler = Labeler_seqgpt()
        else:
            raise ValueError("目前仅支持mt5、seq-gpt")

    def run(self, data, task_type, schema, truth_label = None):
        res = []
        for i in tqdm(range(len(data))):
            input_text = data[i]
            prompt = self.labeler.get_prompt(input_text, task_type, schema)
            prediction = self.labeler.get_prediction(prompt)
            if task_type == 'CLS':
                cls_confidence = self.labeler.get_first_token_confidence(prediction)
                if truth_label is None:
                    res.append({'input_text': input_text, 'Labeler_prediction_with_confidence': [prediction['output']['text'], cls_confidence]})
                else:
                    res.append({'input_text': input_text, 'label': truth_label[i], 'Labeler_prediction_with_confidence': [prediction['output']['text'], cls_confidence], 'is_labeler_right': 'Right' if truth_label[i] == prediction['output']['text'] else 'Wrong'})

            elif task_type == 'NER':
                assert self.model_name in ['seq-gpt']
                ner_with_confidence = self.labeler.get_ner_confidence(prompt+'[GEN]', prediction)
                if truth_label is None:
                    res.append({'input_text': input_text, 'Labeler_prediction_with_confidence': ner_confidence})
                else:
                    res.append({'input_text': input_text, 'label': truth_label[i], 'Labeler_prediction_with_confidence': ner_with_confidence})

        return res
