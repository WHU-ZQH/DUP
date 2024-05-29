# Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems
----

## Requirements
----
* websocket                 0.2.1               
* websocket-client      1.6.1              
* websockets               11.0.3 
* openai 1.3.2

## Running
----
The script for evaluation on GSM8K is shown in "main.py", where the main function is as follows:
```python
def main():
    # model_name = "gpt-4-0613"
    model_name = "gpt-3.5-turbo-0613"
    api_key = "sk-**"  # api——key
    model = ChatGpt(model=model_name, api_key=api_key)
    model.rateLimit = {"RPM":200} 
    model.temperature = 0 
    # prepare datasetss
    sample_num = None 
    Bathch_size = 200
    dataset_inputs, dataset_labels,dataset_answers = gsm8k(sample_num=sample_num, seed=2023, split="test")
    dataset = (dataset_inputs, dataset_labels, dataset_answers)
    if sample_num:
        output_dir = "outputs/gsm8k/test_split_{}_samples".format(sample_num) 
    else:
        output_dir = "outputs/gsm8k/all_test_dataset_gpt3.5" 
    os.makedirs(output_dir, exist_ok=True)
    # save sampled dataset
    if not os.path.exists(os.path.join(output_dir, "dataset.txt")):
        with open(os.path.join(output_dir, "dataset.txt"), "w") as f:
            for data in dataset_inputs:
                f.write(str(data)+"\n")
    if not os.path.exists(os.path.join(output_dir, "labels.txt")):
        with open(os.path.join(output_dir, "labels.txt"), "w") as f:
            for data in dataset_labels:
                f.write(str(data)+"\n")
    if not os.path.exists(os.path.join(output_dir, "answers.txt")):
        with open(os.path.join(output_dir, "answers.txt"), "w") as f:
            for data in dataset_answers:
                f.write(str(data)+"\n")

    # reasoning_base: baseline
    # DUP_prompting: our proposed method
    # DUP_simplified_prompting: our simplified method
    base_acc = reasoning_base(model=model, dataset=dataset, output_dir=output_dir,batch_size=Bathch_size)
    our_acc = DUP_prompting(model=model, dataset=dataset, output_dir=output_dir,Batch_size=Bathch_size)
    our_simplified_acc = DUP_simplified_prompting(model=model, dataset=dataset, output_dir=output_dir,batch_size=Bathch_size)

    print(f"Baseline acc is {base_acc}\nour acc is {our_acc}\nour simplified acc is: {our_simplified_acc}.")
```
If you want to evaluate on the other benchmark, please modify the task type in the "main.py".

## Citation
---
If you find this repository useful, please cite our paper:
```
@article{zhong2024achieving,
  title={Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems},
  author={Zhong, Qihuang and Wang, Kang and Xu, Ziyang and Liu, Juhua and Ding, Liang and Du, Bo and Tao, Dacheng},
  journal={arXiv preprint arXiv:2404.14963},
  year={2024}
}
```