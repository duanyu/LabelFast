## 简单样本，交给LabelFast

LabelFast是中文世界的NLP自动标注开源工具，旨在用LLM技术，快速识别并标注简单文本数据。

使用LabelFast，人类只需关注那些少量而关键的难样本，达到降本增效的效果。

其特点如下：

1. **开箱即用**。无需微调和Prompt工程，提供 标注任务 + 样本，马上开始标注；
    
2. **诚实可信**。在提供标注结果的同时，还提供Confidence信息，以表示模型对标注结果的信心程度，便于使用者确定何时信任模型结果；
    
3. **完全开源**。LabelFast源于开源的模型和技术，因此也将回馈开源社区。  
    

## 版本说明
### **v0.2（最新）**

| 标注任务     | 支持模型 |
| ----------- | ----------- |
| CLS      | mt5、seq-gpt       |
| NER   | seq-gpt        |

1. 标注模型支持：mt5 - [finetuned mT5模型](https://modelscope.cn/models/damo/nlp_mt5_zero-shot-augment_chinese-base/summary "全任务零样本学习-mT5分类增强版-中文-base")、seq-gpt（新增） - [seqgpt-560M模型](https://modelscope.cn/models/iic/nlp_seqgpt-560m/summary);
2. 标注任务支持：CLS - 文本多分类，NER（新增） - 命名实体识别；
3. confidence estimation方法：使用First Token Prob方法。


### **v0.1**
1. 标注模型支持[finetuned mT5模型](https://modelscope.cn/models/damo/nlp_mt5_zero-shot-augment_chinese-base/summary "全任务零样本学习-mT5分类增强版-中文-base")；
2. 标注任务支持文本多分类；
3. confidence estimation使用First Token Prob方法。

## Demo地址

https://modelscope.cn/studios/duanyu/LabelFast/summary

受创空间计算资源限制，Demo**只部署了mt5模型，仅支持CLS任务标注**。

## 如何使用

### 环境依赖

+ python = 3.10
+ 第三方库依赖

``` pip3 install modelscope ms-swift transformers torch scikit-learn sentencepiece ```

### 示例

参照```test.py```

## LabelFast的核心技术

1. **Instruction-Tuning Language Model**。以[Flan-T5](https://arxiv.org/abs/2210.11416 "Flan-T5")、[SeqGPT](https://arxiv.org/abs/2308.10529 "SeqGPT")为代表，基于预训练LLM，在庞大的instruction data（将NLP任务改写为prompt->output的格式）上进行Fine-Tuning，使得模型在NLP任务上具备较强的Zero-Shot Task Generalization能力，能够以Zero-Shot的形式执行众多NLP任务。这部分对应LabelFast中的标注模型。
2. **Confidence Estimation**。得到模型对于标注结果的置信度，目标是尽可能well-calibrated（高confidence -> 高Acc、低confidence -> 低Acc），得到confidence之后，可用于决定何时信任模型标注、何时采用人工标注。计算方法包括Prompting、Entropy、Token Prob等，方法的细节可参照refuel.ai的[这篇博文](https://www.refuel.ai/blog-posts/labeling-with-confidence "refuel.ai blog: labeling with confidence")。

## 联系作者

如果您对LabelFast有任何建议，欢迎添加作者微信进行交流~（VX：duanyu027，最好备注一声“LabelFast”）

如果这个项目对您有帮助，欢迎[Buy Me a Coffee](https://www.buymeacoffee.com/derrick.dy)

