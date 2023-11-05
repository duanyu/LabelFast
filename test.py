from Labeler import Labeler

## 初始化labeler
labeler = Labeler()

## case1：单条/批量测试 without 真实标签
## 直接获取标注结果 + confidence

input_text = ['这件衣服真好看！']
task_type = 'CLS' # CLS - 文本多分类
schema = ['积极', '消极'] # 对于文本分类任务，schema = list of label

res, _ = labeler.run(input_text, task_type, schema)
print(res)

## case2：批量测试 with 真实标签
# 获取标注结果 + confidence + 各个confidence下的任务效果

input_text = ['买的两百多的，不是真货，和真的对比了小一圈！特别不好跟30多元的没区别，退货了！不建议买！',
              '怎么包装盒都没有，就一个塑料袋装了几张面膜，一看就掉了好几个档次了，都不好用了',
              '差到几点的电池！充电一天！用不到一个小时就没电！怎么给小孩带来快乐？',
              '没用几天就坏了。 寄回厂家修理也i没修好。',
              '收到了很实用挺方便的，功能很多爷爷很喜欢呢',
              '去美国游玩>1月以上的朋友应该买一个，很方便。质量也不错。',
              '送货很快，容易安装，东西比想象的好，用起来很方便。']
label = ['消极', '消极', '消极', '消极', '积极', '积极', '积极']
task_type = 'CLS'
schema = ['积极', '消极']

res, threshold_metric = labeler.run(input_text, task_type, schema, label)
print(res)
print(threshold_metric)

