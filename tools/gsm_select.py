import jsonlines
f = open("./outputs/gsm8k_select_cot/whole_test_dataset_3.5_small_dataset/baseline_extracted_responses.txt","r") 
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
i = 0
with jsonlines.open("./data_utils/data/GSM8K/test.jsonl",'r')as f:
    for line in f:
        temp = {"question":str,"answer":str}
        if lines[i][0] =='Y' or lines[i][0] =='y':
            temp["question"] = line["question"] +" Let's think step by step"
        else : temp["question"] = line["question"]
        i = i+1
        temp["answer"] = line["answer"]
        with jsonlines.open("./data_utils/data/GSM8K/small_dataset/test_select_modify.jsonl",'a')as t:
            t.write(temp)