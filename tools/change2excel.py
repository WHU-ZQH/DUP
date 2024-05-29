
import pandas as pd
import os

def change2excel(jsonl_file,output_dir):    
    out_excel = f"{output_dir}/error_question_excel.xlsx"         
    if  os.path.exists(jsonl_file):
               df = pd.read_json(jsonl_file,lines=True)
               df.to_excel(out_excel)

def change2excel_base(jsonl_file,output_dir):    
    out_excel = f"{output_dir}/base_error_question_excel.xlsx"         
    if  os.path.exists(jsonl_file):
               df = pd.read_json(jsonl_file,lines=True)
               df.to_excel(out_excel)

change2excel("./outputs/ablation_GSM8K/error_analysis/useful_error_analysis.jsonl","./outputs/ablation_GSM8K/error_analysis")