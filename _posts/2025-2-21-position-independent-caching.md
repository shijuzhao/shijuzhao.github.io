---
layout: post
title: "位置独立的KV缓存"
theme: jekyll-theme-merlot
date: 2025-02-21
comments: true
---

[论文地址](https://arxiv.org/abs/2502.01960)

TL; DR (Too Long; Didn't Read): KV cache 复用可以不受位置的限制，只要重计算其中小部分即可。

# 研究背景

为了增强多模态大模型(Multimodal Large Language Model, MLLM)的感知能力，处理的图像token 的数量大幅增加，从LLaVA 1.5中的576个增加到LLaVA 1.6中的2304个。Token 数量的显著增长减缓了MLLM推理、限制了应用程序性能。因此所有的主流大模型服务平台如谷歌Gemini、月之暗面Kimi、深度求索DeepSeek、vLLM 和 SGLang都使用前缀缓存的技术来减小首字延迟(Time-to-First-Token, TTFT)。

但是前缀缓存技术仅能复用相同前缀的KV cache。如果两个请求的只在开头的表述不一样，那么整个序列的KV cache都不能被复用。在MLLM的应用场景如图片文本夹杂和多模态检索增强生成(Retrieval-Augmented Generation, RAG)中，前缀缓存是非常低效的。例如，在下图的对话中，如果下一个请求的开头是“We're planning to ...”，那么所有token的KV cache都不能被复用了。而图中的应用场景在互联网上又很常见，如新闻和博客文章都是图片文本夹杂的数据。

![用户对话场景](/Figures/PIC_1.pdf "用户对话场景")
# 论文思路

对于用户输入的处理（也称 Prefill），目前有两种朴素的想法。
- **前缀缓存**：只复用系统提示词和用户输入间的相同部分；
- **完全复用**：直接将多模态信息的KV cache和对应文本的KV cache拼接作推理。这里我们假设用户输入的文本不同而多模态信息相同，因此系统没有新文本的KV cache，只能先计算出它的KV cache再和缓存好的多模态信息的KV cache拼接作Decode。

如下图所示，**前缀缓存**的首字延迟长；**完全复用**违背了大模型计算的Attention 机制，因此生成质量低（这一点在论文中有实验证明）。于是本文提出了部分复用、部分重计算的方法，通过重计算极少量的多模态token，达到减少时延的同时保证生成质量不下降的效果。
![先前工作对比](/Figures/PIC_2.pdf "先前工作对比")

# 关键创新点一：KV cache存储系统设计
我们借鉴了位置独立编码的思想，设计了静态库和动态库来存储多模态信息的KV cache。静态库用于存储用户上传的私有信息，用户之间的数据是隔离的。动态库用于存储互联网上的公开信息以及管理者维护的参考文献。静态库的KV cache连接类似于代码编译的静态链接，它在Prefill阶段（或编译器的编译阶段）将用户输入的文本与图片的KV cache连接起来。动态库的KV cache连接类似于代码执行的动态链接，它是在Decode阶段，当MLLM判断需要调用RAG时，检索器会搜索相关的资料并将它的KV cache与已有的KV cache连接。检索器在这里的作用类似于操作系统中的重定位表。

![InfoBlend系统设计](/Figures/PIC_3.pdf "InfoBlend系统设计")

# 关键创新点二：“选择Attention”机制实现
为了保证生成质量不下降，我们需要选取一些关键token，重计算它们的KV cache，并复用非关键token的KV cache。因此我们实现了选择Attention机制：在每个Attention层中，当关键token的K和V张量被计算出来后，我们用它替换旧的KV cache来进行Attention矩阵的计算。这里我们用了假的KV cache（即全0的张量）来代替未被存储的文本KV cache，因为这假的KV cache在计算Attention前会被新的KV cache替换掉，所以它的取值不重要。

![选择Attention](/Figures/PIC_4.pdf "选择Attention")

# 性能评估
## 主要结果
实验结果显示，InfoBlend节省了54.1%的首字延迟，并在生成质量上没有显著的下降，得分最多下降了13.6%。

![图例](/Figures/PIC_5.pdf "图例")
![主要结果](/Figures/PIC_6.pdf "主要结果")
## 灵敏度分析
当图片的数量变多时，InfoBlend也没有生成质量的显著下降。

![图例](/Figures/PIC_7.pdf "图例")
<div style="display: flex; justify-content: space-around;">
  <div>
    <img src="/Figures/PIC_8.pdf" alt="TTFT" style="max-width:100%;">
  </div>
  <div>
    <img src="/Figures/PIC_9.pdf" alt="Score" style="max-width:100%;">
  </div>
</div>

# 价值与启发
大模型的Attention具有很强的稀疏性，只有少部分token影响大模型的输出。大部分的KV cache都是可以被复用的，这能减少很多计算开销。