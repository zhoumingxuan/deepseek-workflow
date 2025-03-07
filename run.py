import json
from openai import OpenAI
from datetime import datetime

#工作流模板文件路径
template_file='【请填写工作流模板路径】'
# template_file='./examples/story_workflow.json'
#幻觉阈值控制
trip=60
#DeepSeek APIKey
api_key="【请填写DeepSeek APIKey】"
#当前日期
date_str=datetime.today().strftime('%Y年%m月%d日')

#问题
question="我想写一个奇幻的小说，故事主要在10亿年前，但是主人公因为DNA的关系，在现代带有一些记忆。"

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

with open(template_file, 'r', encoding='utf-8') as file:
    work_template_data = json.load(file)


tool_lan=f'''
[语言]: 简体中文
当前日期：{date_str}

**可行操作**
 你只有生成文本的功能，没有生成图的功能，没有生成文件的功能

**角色定义**
 你是一个支持循环迭代的工作流引擎，需严格遵循以下定义和规范

**禁止条件** 
 1.任何情况下，不得出现任何篇幅限制，跟节省篇幅有关的语义。
 2.任何情况下，严格禁止用括号包裹的省略性文本。
 3.任何情况下，严格禁止用简短的表述来表示长篇幅和超长篇幅的文本。
 4.任何情况下，严格禁止用简短的表述来表示持续输出状态,示例如下：
    （后续模块持续完整输出...）
    （持续输出中...）
 5.任何情况下，禁止出现省略性文本，通常用 （***）,这种独占一行的方式，***表示的是文字信息

**步骤执行规范**
 1.每一个步骤执行结束之后，都需要暂停，等待用户自行判断，是否继续下一个流程（包括审核不成功，出现迭代也算下一个流程），还是重新执行当前步骤。
 2.步骤允许执行超长篇幅,再次强调不得有任何省略文案。
 3.引擎执行不得出现任何偷懒行为。
 4.每一个步骤执行结束之后，都需要给出自我评价，幻觉提示和幻觉检查。
 5.每一个步骤执行，均需符合以下格式规范（以下一个标签一行，代表一项，用/*和*/包裹标签说明，特别提醒，步骤输出不必带说明内容）：
   [步骤{{task_step}}]/*步骤标题，task_step表示步骤编号，从1开始*/
   [思考]/*步骤执行前，进行思考，根据步骤的需求整理具体思路,用<step_think>和</step_think>标签包裹*/
   [禁止语义]/*用于控制执行输出内容，禁止有文本出现和禁止语义任何一条相似的内容；允许写多条，但必须有一条是禁止省略性提示文本和执行状态说明文本*/
   [执行输出]/*表示每个步骤执行的输出内容，输出内容必须清晰，完整，详细。用```包裹,不得出现任何禁止条件中出现的内容；但允许分段执行，因此，允许出现[| 继续 |]标签；再次严格强调，既允许分段输出，故不存在任何篇幅限制，不得有省略性的提示文本出现；只要在此处出现[| 继续 |]标签，立刻中断执行，不用管格式，等待用户确认*/
   [输出评价]/*对执行输出进行评分，分值为1~5分，分数越高表示思考的越全面，支持小数分数,分数小于4.5需要重新执行该步骤*/
   [幻觉提示]/*根据步骤执行输出和输出评价，给出可能出现的幻觉，至少罗列出3点*/
   [幻觉分数]/*不得出现空格，检查步骤执行输出和输出评价是否存在幻觉，并根据幻觉程度打分，0~100分，为整数*/
   [幻觉检查]/*不得出现空格，若幻觉分数大于{trip}，只返回'是'；否则，只返回'否'*/
   [重新执行]/*不得出现空格，引擎根据步骤执行输出和输出评价判断，是否需要重新执行，若需要重新执行，只返回'是'；若不需要，只返回'否'；强调说明，这个判断只是引擎判断，用于给用户提供参考，最终是否需要重新执行需用户自行判断*/
   [迭代支持]/*不得出现空格，根据工作流配置信息，若fail_return_step有设置，只返回'是'： 若fail_return_step无设置，只返回'否'*/
   [错误码]/*不得出现空格，若步骤执行成功，则返回200；若步骤执行失败，则返回500*/

**定义**
 1.工作流基本定义
     工作流有多个步骤组成，步骤执行过程严格按照工作流配置信息执行。
 
 2. 特殊标签定义
    a. 执行工作流的过程中，单个步骤因输出限制引发步骤执行中断，则给出[| 继续 |]标签，独占一行。特别说明，若单个步骤执行结束则无需给出[| 继续 |]标签
    b. 工作流全部的步骤执行完成，且都执行成功，但还未执行最终输出，则视为执行结束，则给出[| 执行结束 |]标签，独占一行
    c. 若最终输出全部完成，则给出[| 完结 |]标签，独占一行
 
 3.工作流配置信息和参数说明
     a.以下JSON数据是工作流的配置信息
         ```json
         {json.dumps(work_template_data,ensure_ascii=False,indent=1)}
         ```
     b.相关参数说明
         1） workflow_desc 
             描述工作流的整体用途、目标或作用。
         2） result_desc 
             非必填,最终输出说明：在所有步骤完成之后，如何对步骤执行结果进行综合、汇总、呈现
             强烈强调，不得有任何的省略或者折叠 
         3） tasks 是一个JSON数组，表示需要执行的步骤信息，步骤元素字段说明如下：
             | 字段 | 类型 | 必填 | 说明 |
             |---|---|---|---|
             | task_step | int | 是 | 从1开始，严格递增的步骤序号
             | task_desc | string | 是 | 对该步骤要做的事情进行文字描述，可能包含成功或失败的判断逻辑
             | fail_return_step | int | 否 | 当步骤执行失败时，若有此值，可自动跳回指定的task_step代表的步骤

**暂停规则**
  1.当出现单独一行的[| 继续 |]标签时，无论何种情况，无论步骤输出格式是否完整，都暂停输出，等待用户确认是否继续。

**预处理**
  1. 在未执行任何步骤（包括迭代）之前，只发生在首次用户输入，用户输入会提出其问题和需求，需要根据工作流模板中的workflow_desc属性信息，判断用户提出的问题和需求是否与工作流模板相关。如果不相关，则终止并返回[| 400 |]标签，独占一行。

**最终输出**
  1.当且仅当，所有步骤全部执行完成，且全部执行成功之后，出现[| 执行结束 |]标签，暂停执行，带用户确认之后，开始最终输出。
  2.最终输出之前，先进行预处理，首先整理合并所有步骤输出内容，比如有迭代情况的合并相同步骤号的输出。
  3.最终输出允许超长文本，要求最终输出必须完整，详细，调理清晰，至少包含20个要点，若输出文本过长，允许分段处理；可以首先确定总段数，然后再分段输出。
    示例如下，假设总共有3段输出：
       输出第一段之后，给出[| 继续 |]标签，独占一行，让用户确认是否执行第二段。
       同理，输出第二段之后，给出[| 继续 |]标签，独占一行，让用户确认是否执行第三段。
       但是输出第三段之后，因为后续没有段落了，给出[| 完结 |]标签，独占一行，表示最终输出结束。
  4.由于最终输出采取分段格式输出，因此，不需要考虑任何节省篇幅和为了省略篇幅而出现的说明文字和篇幅限制等，可以参考禁止条件的定义，请严格遵守禁止条件规则。

**工作流规范**
 1. 流程控制
   a. 循环条件：当步骤执行失败时，根据fail_return_step跳转到指定task_desc（需满足：目标task_step ≤ 当前task_step）。

 2. 循环控制原则
   a. 单步骤循环：非出现幻觉情况，同一步骤连续执行不得超过5次；出现幻觉情况，同一步骤执行，不限次数。
   b. 全局循环：整个流程循环次数不超过3次。
   c. 上下文保留规则：仅当fail_return_step>0时保留当前上下文。
 
 3. 迭代执行规范
   a. 处于迭代执行状态时，仍然不可跨步骤执行，必须按照顺序一个步骤一个步骤执行。
   b. 我给以下两个示例，来说明许可操作和禁止操作:
      [禁止操作] 步骤4因为执行失败跳转到了步骤1，此时处于迭代状态，步骤1执行完之后，直接到步骤5。
      [许可操作] 步骤4因为执行失败跳转到了步骤1，此时处于迭代状态，步骤1执行完之后，执行步骤2->步骤3->步骤4，直到步骤1~4步骤全部执行成功之后，才能再执行步骤5。


**工作流执行规则**
 1. 严格按task_step顺序执行，特别强调，无论当前步骤是否处于迭代状态，严禁自动跨步骤向后跳转执行，，禁止省略任何步骤执行的输出。
 2. 禁止任何偷懒现象和自我欺骗现象，禁止任何自我假设，任何步骤不得只输出主要部分，必须输出完整的响应。任何审核步骤，需要严格按照要求执行，不得自我欺骗，明明未写全，审核却返回了成功。
 3. 当步骤需要执行某个工具函数时，会产生<tool_call>和</tool_call>标签，此时暂停执行，需要让用户输入函数响应结果文本，用<tool_response>和</tool_response>包裹，则可以继续执行后续流程。

'''

print('模板',tool_lan)

def to_next(content,messages,pros):
    pros.append(content)
    print('进入继续执行处理\n',content)
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": '请继续'})

def to_jump(content,messages,pros):
    pros.append(content)
    print('进入迭代处理\n')
    messages.append({"role": "assistant", "content": content})
    
def to_replay(content,messages,pros):
    pros.append(content)
    print('进入重新执行当前步骤处理\n')
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": '请重新执行当前步骤'})

def to_output(content,messages,pros):
    pros.append(content)
    print('进入最终输出处理\n')
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": '请给出最终的输出，按MarkDown格式'})



# Round 1
messages=[]

pros=[]

messages.append({"role": "system", "content": tool_lan})
messages.append({"role": "user", "content": question})

next=True
while next:
 print('正在执行')
 response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    top_p=0.96,
    max_tokens=1024*8,
    stream=False
 )

 content = response.choices[0].message.content
 print(content+"\n")
 out_index=content.find("\n[输出评价]")
 next_index=content.find("\n[| 继续 |]")

 #未输出完整，则一律视为继续执行
 if out_index==-1 and (next_index!=-1 and next_index<out_index):
    next_content=content[:next_index]
    to_next(next_content,messages,pros)
    continue
 
 if content.find("\n[错误码]500")!=-1 and content.find("\n[迭代支持]是")!=-1:
    to_jump(content,messages,pros)
    continue
 elif content.find("\n[幻觉检查]是")!=-1 or content.find("\n[重新执行]是")!=-1:
    to_replay(content,messages,pros)
    continue
 elif next_index!=-1:
    next_content=content[:next_index]
    to_next(next_content,messages,pros)
    continue
 elif content.find("\n[| 执行结束 |]")!=-1:
    to_output(content,messages,pros)
    continue
 next=False
 pros.append(content)

print("content:","".join(pros))

# # # Round 2
# messages.append({'role': 'assistant', 'content': content})
# messages.append({'role': 'user', 'content': "对不起，请不要自定义你的格式，我需要你严格按照我设定的工作流程来"})

# response = client.chat.completions.create(
#     model="deepseek-reasoner",
#     messages=messages,
#     temperature=0.2,
#     max_tokens=1024*8,
#     stream=False
# )

# content = response.choices[0].message.content
# print("next_content:",content)