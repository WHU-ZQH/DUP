import os
import json
import jsonlines
import random

def gsm8k(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/GSM8K/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            print(line["question"])
            answer = line["answer"].split("####")
            label = answer[1].strip()
            labels.append(label)
            answers.append(line["answer"])
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

def math_3(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/MATH/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["problem"])
            # print(line["question"])
            answer = line["answer"]
            label = answer.strip()
            labels.append(label)
            answers.append(line["answer"])
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

def SVAMP(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/SVAMP/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

def AddSub(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/AddSub/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

def MultiArith(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/MultiArith/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)
    
def AQuA_nochoice(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join('./data_utils/data/origin_dataset/AQuA.json')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []
    options = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["correct"])
            options.append(line["options"])
            label = answer
            labels.append(label)
            answers.append(str(line["correct"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers,options
    return (inputs,labels,answers,options)
    
def AQuA(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/AQuA/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)


def SingleEq(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/SingleEq/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

    
def CommonsenseQA(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/CommonsenseQA/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

   
def coin_flip(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/coin_flip/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

   
def last_letters(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/last_letters/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

def StrategyQA(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/StrategyQA/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)

def FOLIO(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/FOLIO/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["question"])
            answer = str(line["answer"])
            label = answer
            labels.append(label)
            answers.append(str(line["answer"]))
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)


def race(sample_num=None, seed=2023, split="test"):
    random.seed(seed)
    current_file_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(current_file_path), f'data/GSM8K/{split}.jsonl')
    assert os.path.exists(data_path)
    inputs = []
    labels = []
    answers = []

    with jsonlines.open(data_path,'r')as f :
        for line in f:
            inputs.append(line["article"])
            answer = 0
            label = 0
            labels.append(label)
            answers.append(0)
    
    if sample_num:
        sampled_inputs,sampled_labels = zip(*random.sample(list(zip(inputs,labels)), sample_num))
        return sampled_inputs,sampled_labels,answers
    return (inputs,labels,answers)