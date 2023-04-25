from transformers import pipeline
question_answerer = pipeline("question-answering")
context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
"""
response = question_answerer(question="What is extractive question answering?", context=context)
print(response)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')

zh_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
QA_input = {'question': "著名诗歌《假如生活欺骗了你》的作者是",'context': "普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。"}
response = zh_qa(QA_input)
print(response)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
sentence = "弱小的我也有大梦想"
tokens = tokenizer.tokenize(sentence)
print(tokens)

print(len(tokenizer.vocab))

ids = tokenizer.encode(sentence)
print(ids)

ids = tokenizer.encode(sentence, padding="max_length", max_length = 15)
print(ids)

attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)
print(attention_mask, token_type_ids)

inputs = tokenizer(sentence, padding="max_length", max_length = 15)
print(inputs)

sentences = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵",
        "至少我们还有诗歌"]
response = tokenizer(sentences, padding="max_length", max_length=15)
print(response)

from datasets import load_dataset
datasets = load_dataset("madao33/new-title-chinese")
print(datasets)

from datasets import list_datasets
print(list_datasets()[:20])


