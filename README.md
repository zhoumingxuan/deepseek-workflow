# 引言
   ## 简介
     本项目是基于DeepSeek模型用提示词去实现自定义工作流，从而实现ai自动化或半自动化流程。适用于蒸馏模型和全参数模型，deepseek_api，建议采用全参数模型或者deepseek_api，经过一些实验，蒸馏14B模型，虽然能安装流程执行，但因为参数量相对较小，回答会出现幻觉，也无法出现多次迭代的能力。
   ## 知识介绍
     提示词是用户输入给大模型的指令或问题，用于引导模型生成特定输出。这种方式是利用了自回归模型中，根据上下文推断出下一个词的原理。提示词分为很多种，指令式，角色扮演，示例驱动，链式思考，对抗提示。而我所做的自定义工作流，更像是一种引导式思维链，通过逐个步骤，一步一步生成内容，从而得到更详细更准确的答案。
     而在这流程基础之上，我还做了更进一步的操作，通过步骤指令式的实现让ai自我审核(类似ai的一种自我反馈机制）使得生成的文本，更为丰富，更为准确。
