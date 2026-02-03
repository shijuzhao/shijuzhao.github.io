---
layout: post
title: "原生位置独立缓存"
theme: jekyll-theme-merlot
date: 2026-02-03
comments: true
permalink: /npic
repository_url: https://github.com/shijuzhao/Comb
---

[论文地址](https://arxiv.org/abs/2602.01519)

TL; DR (Too Long; Didn't Read): 我们把原始Transformer结构的Encoder装回来了，赋予了大模型原生位置独立缓存的能力。

# 研究背景

当前大模型的KV cache都是基于前缀的（因为要经过位置编码），如果前缀不相同，KV cache就不能复用，否则模型将会胡言乱语。这个限制使得在检索增强生成(Retrieval-Augmented Generation, RAG)的场景下，模型推理十分低效。因为检索回来的多个文档以任意可能的顺序排列，排列在后面的文档的KV cache不能复用，只能重新计算。

为了应对这个限制，位置独立缓存(Position-Independent Caching, PIC)被提出，其流程类似于代码的编译与链接，如下图所示：
1. 对多个文档块分别进行编译，生成各自的KV cache；
2. 存储KV cache到KVLib，需要复用该文档时再将其取出；
3. 将位置独立的KV cache链接起来，用于大模型新的推理。

![PIC流程](/Figures/NPIC_1.pdf "PIC流程")
# 论文思路

目前PIC没被业界使用的原因是它会导致模型生成质量下降，无法恢复到原来的水平。现有的PIC技术可以被分为两类：
1. **无需训练的(post-training)**：它不改变模型的结构以及参数，在链接阶段通过选择少数几个token来重计算的方法恢复一定的准确度。这种方法不能保证模型准确度不下降。
2. **基于微调的(training-aware)**：为了避免模型准确度下降，对模型参数在分块注意力掩码的场景下进行微调，使其适应位置独立的KV cache。但这种方法修改了模型的原有参数，可能会导致灾难性遗忘以及在原始非RAG任务的准确度下降。

为了解决两类方法的限制，又结合它们的优点，我们提出了原生位置独立缓存(native PIC)，不改变模型参数与原有结构，而是在decoder-only的大模型上外置一个encoder插件，并通过监督微调的方式来训练encoder使其可以理解位置独立的文档。
1. **基于训练**：可以解决模型准确度下降的问题。
2. **非侵入式**：如果用户不想使用PIC功能，可以将所有文本放入decoder的输入，不改变模型结果（因为decoder的模型参数不变）。
3. **高效推理**：尽管我们添加了新参数，但模型的计算量并没有增加。

# 关键创新点一：COMB缓存系统设计
COMB是一个专为管理位置独立的KV cache（简称PICache）而设计的缓存系统，可以与主流大模型推理系统如transformers、vLLM和SGLang集成。

输入给COMB系统的请求包含一个问题和多个文档。这里我们让用户来区分哪些是问题，哪些是希望被复用KV cache的文档，因为用户最清楚他们自己的需求。首先，COMB用一个哈希表查看是否存在这些文档的PICache，否则让chunk processor去计算PICache，存储PICache并更新哈希表。然后COMB将这些PICache链接起来发给推理引擎，推理引擎根据问题和PICache返回应答消息。

![COMB系统设计](/Figures/NPIC_2.png "COMB系统设计")

# 关键创新点二：Encoder插件设计
我们在decoder-only的大模型上加入了encoder。如图所示，decoder的参数保持冻结，负责文本生成，问题和新生成的token都会进入decoder；encoder的参数通过监督微调确定，负责生成文档的PICache，将这些PICache存储起来以复用。encoder与decoder通过交叉注意力模块交互，decoder由此获知上下文信息。

![模型设计](/Figures/NPIC_3.png "模型设计")

# 性能评估
## 主要结果
实验结果显示，COMB节省了51-94%的首字延迟，并拥有相仿的准确度。在Deepseek-V2-Lite-Chat模型的测试上，COMB甚至超出了基座模型的准确率，这是因为CombDeepseek是由更强大的Llama-3.1-8B-Instruct模型生成的数据训练的。

![Llama](/Figures/NPIC_4.png "Llama")

![Deepseek](/Figures/NPIC_5.png "Deepseed")

## 吞吐量
随着请求到达速度的增加，请求处理的首字延迟和吞吐量增加。但COMB保持了最低的首字延迟和最高的吞吐量，与业界生产环境系统相比提高了3倍吞吐量。

![吞吐量](/Figures/NPIC_6.png "吞吐量")

# 价值与启发
如今我们已经进入了Agent的时代。什么是Agent, 它与LLM有什么区别？一个普遍认可的说法是Agent可以根据长期目标制定短期动作，会调用工具。那么Agent一个很重要的能力就是检索。除了需要查阅参考文献来获得信息，Agent也需要检索应该使用哪个工具更合适。只要涉及到检索，就需要PIC，因为检索回来的东西在各种排列组合的可能下一定是乱序的。如果使用前缀缓存，基本上只有排在第一个的项目可以复用KV cache，后面的KV cache全都不能复用，这非常低效。

我们可以给Agent装上一个Encoder，将所有检索到的乱序的东西丢进Encoder，问题和模型的思考留在Decoder，当模型需要检索新的东西时，就把以前检索到的东西丢掉，把新检索到的东西扔进Encoder。想象一下，如果Decoder的128K上下文里只有问题和模型自己的思维链，没有嘈杂的参考文献的信息，那么这个Agent的能力将会有多强。
