import os
import jsonlines

import pandas as pd
def analysis_base_aug_base_aug_cot(base_file,base_new_file,num=1318):
    base_new = []
    base = []
    with jsonlines.open(base_new_file,'r')as f:
        for line in f:
            base_new.append(line["idx"])
    with jsonlines.open(base_file,'r')as f:
        for line in f:
            base.append(line["idx"])
            
    both_correct = []
    base_new_error = base_new.copy()
    base_error = base.copy()
    
    correct = [i+1 for i in range(num)]
    correct_base = correct.copy()
    correct_new = correct.copy()
    for i in range(len(base_new_error)):
        correct_new.remove(base_new_error[i])
    
    for i in range(len(base_error)):
        correct_base.remove(base_error[i])
     
     
    correct_aug_cot =  correct_new.copy()
    correct_aug = correct_base.copy()   

    for i in range(len(correct_new)):
        for j in range(len(correct_base)):
            if correct_new[i] == correct_base[j]:
                both_correct.append(correct_new[i])
                # a cot 
                correct_aug_cot.remove(correct_new[i])
                # b base
                correct_aug.remove(correct_base[j])
    

   
    return correct_aug_cot,correct_aug

def analysis_base_base_aug_cot(base_file,base_new_file,num=1318):
    base_new = []
    base = []
    with jsonlines.open(base_new_file,'r')as f:
        for line in f:
            base_new.append(line["idx"])
    with jsonlines.open(base_file,'r')as f:
        for line in f:
            base.append(line["idx"])
            
    both_correct = []
    base_new_error = base_new.copy()
    base_error = base.copy()
    
    correct = [i+1 for i in range(num)]
    correct_base = correct.copy()
    correct_new = correct.copy()
    for i in range(len(base_new_error)):
        correct_new.remove(base_new_error[i])
    
    for i in range(len(base_error)):
        correct_base.remove(base_error[i])
     
     
    correct_aug_cot =  correct_new.copy()
    correct_b = correct_base.copy()   

    for i in range(len(correct_new)):
        for j in range(len(correct_base)):
            if correct_new[i] == correct_base[j]:
                both_correct.append(correct_new[i])
                # a cot 
                correct_aug_cot.remove(correct_new[i])
                # b base
                correct_b.remove(correct_base[j])
    


    return correct_aug_cot,correct_b

def analysis_base_cot(base_file,base_new_file,num = 299):
    base_new = []
    base = []
    with jsonlines.open(base_new_file,'r')as f:
        for line in f:
            base_new.append(line["idx"])
    with jsonlines.open(base_file,'r')as f:
        for line in f:
            base.append(line["idx"])
            
    both_correct = []
    base_new_error = base_new.copy()
    base_error = base.copy()
    
    correct = [i+1 for i in range(num)]
    correct_base = correct.copy()
    correct_new = correct.copy()
    for i in range(len(base_new_error)):
        correct_new.remove(base_new_error[i])
    
    for i in range(len(base_error)):
        correct_base.remove(base_error[i])
     
     
    correct_cot =  correct_new.copy()
    correct_b = correct_base.copy()   

    for i in range(len(correct_new)):
        for j in range(len(correct_base)):
            if correct_new[i] == correct_base[j]:
                both_correct.append(correct_new[i])
                # a cot 
                correct_cot.remove(correct_new[i])
                # b base
                correct_b.remove(correct_base[j])
    

   
    return correct_cot,correct_b
    
    
    
def analysis_base_aug(base_file,base_new_file, num=1318):
    base_new = []
    base = []
    with jsonlines.open(base_new_file,'r')as f:
        for line in f:
            base_new.append(line["idx"])
    with jsonlines.open(base_file,'r')as f:
        for line in f:
            base.append(line["idx"])
            
    both_correct = []
    base_new_error = base_new.copy()
    base_error = base.copy()
    
    correct = [i+1 for i in range(num)]
    correct_base = correct.copy()
    correct_new = correct.copy()
    for i in range(len(base_new_error)):
        correct_new.remove(base_new_error[i])
    
    for i in range(len(base_error)):
        correct_base.remove(base_error[i])
     
     
    correct_aug =  correct_new.copy()
    correct_b = correct_base.copy()   

    for i in range(len(correct_new)):
        for j in range(len(correct_base)):
            if correct_new[i] == correct_base[j]:
                both_correct.append(correct_new[i])
                # a cot 
                correct_aug.remove(correct_new[i])
                # b base
                correct_b.remove(correct_base[j])

    return correct_aug,correct_b
    
def analysis_basecot_baseaug(base_file,base_new_file, num=1318):
    base_new = []
    base = []
    with jsonlines.open(base_new_file,'r')as f:
        for line in f:
            base_new.append(line["idx"])
    with jsonlines.open(base_file,'r')as f:
        for line in f:
            base.append(line["idx"])
            
    both_correct = []
    base_new_error = base_new.copy()
    base_error = base.copy()
    
    correct = [i+1 for i in range(num)]
    correct_base = correct.copy()
    correct_new = correct.copy()
    for i in range(len(base_new_error)):
        correct_new.remove(base_new_error[i])
    
    for i in range(len(base_error)):
        correct_base.remove(base_error[i])
     
     
    correct_aug =  correct_new.copy()
    correct_cot = correct_base.copy()   

    for i in range(len(correct_new)):
        for j in range(len(correct_base)):
            if correct_new[i] == correct_base[j]:
                both_correct.append(correct_new[i])
                # a cot 
                correct_aug.remove(correct_new[i])
                # b base
                correct_cot.remove(correct_base[j])

    return correct_aug,correct_cot
            
def aug_cot_analysis(aug,cot):
    com_correct = []
    aug_correct =aug.copy()
    cot_correct = cot.copy()
    for i in range(len(aug)):
        for j in range(len(cot)):
            if aug[i] == cot[j]:
                com_correct.append(aug[i])
                aug_correct.remove(aug[i])
                cot_correct.remove(cot[j])
    return aug_correct,cot_correct

def aug_cot_analysis_correct(aug,cot):
    com_correct = []
    aug_correct =aug.copy()
    cot_correct = cot.copy()
    for i in range(len(aug)):
        for j in range(len(cot)):
            if aug[i] == cot[j]:
                com_correct.append(aug[i])
                aug_correct.remove(aug[i])
                cot_correct.remove(cot[j])
    return aug_correct,cot_correct,  com_correct      

def generate_excel(error_file,file_name,com_correct):
    i = 0
    with jsonlines.open(error_file,'r')as f:
        for line in f:
            if i < len(com_correct) and line["idx"] == com_correct[i] :
                i = i +1
                with jsonlines.open(file_name,'a') as p:
                    p.write(line)

    file = file_name.split(".json")
    excel = file+".xlsx"
    df = pd.read_json(file_name,lines=True)
    df.to_excel(excel)
    
    
def main():
    num_dataset=1250
    
    cot , base_cot =  analysis_base_cot("./outputs/gsm8k_select_cot/whole_test_dataset_3.5_judge/base_error_questions.jsonl",
                                    "./outputs/gsm8k_select_cot/whole_test_dataset_3.5_small_cot/base_error_questions.jsonl", num=num_dataset)   
    
    aug_com , cot_com =  analysis_basecot_baseaug("./outputs/gsm8k/test_split_1250_samples_base_cot_79.36/base_error_questions.jsonl",
                                    "./outputs/gsm8k/test_split_1250_samples_75.84-81.52/aug_error_questions.jsonl", num=num_dataset)  
    aug_correct_base,cot_correct_base = aug_cot_analysis(base_aug,base_cot)
    aug_correct,cot_correct = aug_cot_analysis(aug,cot)
    cot_correct_agu_error,base_correct_agu_error,com_correct = aug_cot_analysis_correct(base_aug,cot_com)
    
    generate_excel("./outputs/gsm8k/test_split_1250_samples_75.84-81.52/aug_error_questions.jsonl",
                   "./outputs/gsm8k/test_split_1250_samples_75.84-81.52/base_and_cot_error_questions.jsonl",com_correct)
    
    print("aug_correct+base correct, but ours fail"+",".join(str(com_correct[i]) for i in range(len(com_correct))))
  
if __name__ =="__main__":
    main()
        
            