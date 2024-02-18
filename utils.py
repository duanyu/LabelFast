import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def get_cls_performance_by_threshold(res, threshold):
    # 得到分类任务下，某个threshold下的metric情况，根据需要选择合适的threshold
    threshold_table = []

    new_res = [d for d in res if d['Labeler_prediction_with_confidence'][1] >= threshold]
    coverage = len(new_res) / len(res)
    truth, pred = [d['label'] for d in new_res], [d['Labeler_prediction_with_confidence'][0] for d in new_res]
    acc = accuracy_score(truth, pred)
    macro_f1 = f1_score(truth, pred, average='macro')
    micro_f1 = f1_score(truth, pred, average='micro')
    threshold_table.append({'confidence_threshold': threshold, 'acc': acc, 'coverage': coverage, 'macro_f1': macro_f1, 'micro_f1': micro_f1})

    return threshold_table

def get_ner_performance_by_threshold(res, threshold):
    # 得到ner任务下，某个threshold下的metric情况，根据需要选择合适的threshold
    threshold_table = []

    # 第一步：基于threshold，得到truth，pred
    truth_pred_list = []
    for single_res in res:
        new_pred = {}
        for entity_type in single_res['Labeler_prediction_with_confidence']:
            entities = [e[0] for e in single_res['Labeler_prediction_with_confidence'][entity_type] if e[1] >= threshold]
            new_pred[entity_type] = entities
        truth_pred_list.append([single_res['label'], new_pred])

    # 第二步：基于ner_pred和ner_ground truth，得到指标（precision\recall\macroF1\microF1)
    entity_metric = {}
    for entity_type in schema:
        entity_metric[entity_type] = {'tp':0, 'fp':0, 'fn':0}

    for truth_pred in truth_pred_list:
        ner_truth, ner_pred = truth_pred[0], truth_pred[1]
        for entity_type in schema:
            tp, fp, fn = 0, 0, 0
            pred = ner_pred.get(entity_type)
            truth = ner_truth.get(entity_type)
            if pred is None and truth is None:
                pass
            elif pred is None and truth is not None:
                fn += len(truth)
            elif pred is not None and truth is None:
                fp += len(pred)
            else:
                pred = set(pred)
                truth = set(truth)
                tp += len(pred & truth)
                fp += (len(pred) - tp)
                fn += (len(truth) - tp)

            entity_metric[entity_type]['tp'] += tp
            entity_metric[entity_type]['fp'] += fp
            entity_metric[entity_type]['fn'] += fn

    p = sum([entity_metric[et]['tp'] for et in entity_metric]) / (sum([entity_metric[et]['tp'] for et in entity_metric]) + sum([entity_metric[et]['fp'] for et in entity_metric]))
    r = sum([entity_metric[et]['tp'] for et in entity_metric]) / (sum([entity_metric[et]['tp'] for et in entity_metric]) + sum([entity_metric[et]['fn'] for et in entity_metric]))
    f1 = 2*p*r/(p+r)

    entity_p, entity_r, entity_f1 = {}, {}, {}
    for et in entity_metric:
        tp, fp, fn = entity_metric[et]['tp'], entity_metric[et]['fp'], entity_metric[et]['fn']
        if (tp+fp) == 0:
            entity_p[et] = None
        else:
            entity_p[et] = tp/(tp+fp)

        if (tp+fn) == 0:
            entity_r[et] = None
        else:
            entity_r[et] = tp/(tp+fn)

        if entity_p[et] is not None and entity_r[et] is not None:
            entity_f1[et] =  2 * entity_p[et] * entity_r[et]/ (entity_p[et]+ entity_r[et])
        else:
            entity_f1[et] = None

    macro_p = np.mean([v for v in entity_p.values() if v is not None])
    macro_r = np.mean([v for v in entity_r.values() if v is not None])
    macro_f1 = np.mean([v for v in entity_f1.values() if v is not None])

    threshold_table.append({'confidence_threshold': threshold, 'micro_p': p, 'micro_r': r, 'micro_f1': f1, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1, 'entity_p': entity_p, 'entity_r': entity_r, 'entity_f1': entity_f1})

    return threshold_table
