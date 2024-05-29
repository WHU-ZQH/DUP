import jsonlines
st = """ /nPlease determine whether the above questions need to be added "Let's think step by step",If the question satisfies one of the following conditions:/n-. the question description is complex or difficult to understand./n-. requires additional calculations and steps./n-. There are many numbers and variables present./n-. the quetiong have more logical correlations./njust need to output: Yes,otherwise output: No, show the answer and give reasons!"""
with jsonlines.open("./data_utils/data/GSM8K/small_dataset/test.jsonl",'r')as f:
    for line in f:
        temp = {"question":str,"answer":str}
       
        temp["question"] = line["question"] +st
        
        
        temp["answer"] = line["answer"]
        with jsonlines.open("./data_utils/data/GSM8K/small_dataset/test_select_cot.jsonl",'a')as t:
            t.write(temp)