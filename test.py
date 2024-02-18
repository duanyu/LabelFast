from Labeler import Labeler
from utils import get_cls_performance_by_threshold, get_ner_performance_by_threshold

## 初始化labeler
labeler = Labeler(model_name = 'seq-gpt') #v0.2版支持 mt5、seq-gpt 两种模型

## case1：单条/批量测试 without 真实标签
## 直接获取标注结果 + confidence

# 分类
input_text = ['我好喜欢这件衣服']
task_type = 'CLS' # CLS - 文本多分类
schema = ['积极', '消极'] # 对于文本分类任务，schema = list of label

res = labeler.run(input_text, task_type, schema)
print(res)

# 抽取
input_text = ['赛后公牛队主教练杰克逊对罗德曼的表现大加赞赏。']
task_type = 'NER' # NER - 命名实体识别
schema = ['地点', '人物'] # list of entity type

res = labeler.run(input_text, task_type, schema)
print(res)

## case2：单条/批量测试 with 真实标签
## 给定threshold，获取标注结果 + confidence + 该threhold下的metric，帮助挑选合适的threshold

# 分类
input_text = ['买的两百多的，不是真货，和真的对比了小一圈！特别不好跟30多元的没区别，退货了！不建议买！',
              '怎么包装盒都没有，就一个塑料袋装了几张面膜，一看就掉了好几个档次了，都不好用了',
              '差到几点的电池！充电一天！用不到一个小时就没电！怎么给小孩带来快乐？',
              '没用几天就坏了。 寄回厂家修理也i没修好。',
              '收到了很实用挺方便的，功能很多爷爷很喜欢呢',
              '去美国游玩>1月以上的朋友应该买一个，很方便。质量也不错。',
              '送货很快，容易安装，东西比想象的好，用起来很方便。']
ground_truth = ['消极', '消极', '消极', '消极', '积极', '积极', '积极']
task_type = 'CLS'
schema = ['积极', '消极']

res = labeler.run(input_text, task_type, schema, ground_truth)
print(res)
for threshold in [0.5, 0.75, 0.9]:
    threshold_table = get_cls_performance_by_threshold(res, threshold)
    print(threshold_table)

# 抽取
input_text = ['赛后公牛队主教练杰克逊对罗德曼的表现大加赞赏。',
              '罗莎琳在洛杉矶举办了一场慈善音乐会，为当地无家可归者筹集了数百万元。',
              '法国总统马克龙与德国总理默克尔在柏林举行了一场高级别会议，讨论了欧盟未来的发展方向。']
task_type = 'NER'
schema = ['地点', '人物']
ground_truth = [{'人物': ['杰克逊', '罗德曼']},
                {'地点': ['洛杉矶'], '人物': ['罗莎琳']},
                {'地点': ['柏林'], '人物': ['马克龙', '默克尔']}]

res = labeler.run(input_text, task_type, schema, ground_truth)
print(res)
for threshold in [0.5, 0.6, 0.7]:
    threshold_table = get_ner_performance_by_threshold(schema, res, threshold)
    print(threshold_table)
