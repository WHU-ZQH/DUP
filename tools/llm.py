import ast
import jsonlines
import pprint
from abc import ABC, abstractmethod
import requests
import json
from datetime import datetime
import time
import subprocess
import tqdm
import openai
import itertools
import argparse
import threading
import shutil
import _thread as thread
import base64
import datetime
import hashlib
import hmac
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import linecache
import jsonlines
import random
import os
import backoff
import requests
from requests.auth import HTTPBasicAuth

class BaseModel(ABC):
    def __init__(self, rateLimit={}, ) -> None:
        """
        rateLimit setting
        """
        super().__init__()

        self.thread_pools = []

        self.rateLimit = rateLimit

        if "RPM" in self.rateLimit:
            # RPM : the request rate limit per minute
            # init time for Rate Limit
            self._minute = datetime.now().minute
            self._avail_req = self.rateLimit["RPM"]
        elif "SPR" in self.rateLimit:
            # SPR : second per request (low speed request)
            self._lastTime = datetime.now()


    def configure_params(self, temperature=1.0, top_p=0.95, top_k=40, max_output_tokens=1024, candidate_count=1, repetition_penalty=1, stream=False):
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.max_output_tokens=max_output_tokens
        self.candidate_count = candidate_count
        self.repetition_penalty = repetition_penalty
        self.stream = stream


    @abstractmethod
    def generate(self, prompt):
        pass
    

    def generate_with_price(self, prompt):
        """return both the response of prompt and its cost.

        Args:
            prompt (str): the prompt passed into the api.
        """
        return self.generate(prompt), 0.1
        
        
    def estimate_cost(self, input_file, sample_num=100):
        """For given dataset, esimate it cost for specifically api call.

        Args:
            input_file (str): abs path of the input file.
            sample_num (int): the sample number in the estimation. Defaults to 100.
        """
        # first get the sampled dataset
        result = subprocess.run(["wc", "-l", input_file], capture_output=True, text=True)
        dataset_num = int(result.stdout.strip().split()[0])

        if sample_num >= dataset_num:
            sample_num = dataset_num

        print("***** dataset number *** sample number  ***")
        print(f"***** {dataset_num} \t\t *** {sample_num} \t\t ***")

        progress_bar = tqdm.tqdm(total=sample_num)

        random.seed(100)
        sampled_indices = random.sample(range(dataset_num), sample_num)
        sampled_dataset = []
        with jsonlines.open(input_file) as reader:
            for idx, line in enumerate(reader):
                if idx in sampled_indices:
                    sampled_dataset.append(line)

        try:
            sampled_data = [line["prompt"] for line in sampled_dataset]
        except:
            print("not find prompt key in data file")
            exit()
        

        lock = threading.Lock()

        output_res = []
        # call api
        for index, data in enumerate(sampled_data):    
            while True:
                lock.acquire()
                rateLimit = self.isRateLimited()
                lock.release()
                if rateLimit:
                    time.sleep(1)
                else:
                    break
            def _my_function(data, output_res, progress_bar, lock):
                max_req = 10
                req_time = 0
                while req_time<max_req:
                    req_time += 1
                    try:
                        response, price = self.generate_with_price(data)
                        if response!="<error>":
                            break
                    except:
                        while True:
                            lock.acquire()
                            rateLimit = self.isRateLimited()
                            lock.release()
                            if rateLimit:
                                time.sleep(1)
                            else:
                                break

                if req_time >= max_req:
                    print("Intnet Error!")
                    response = "<error>"
                    price = 0
                
                if response in ["", None]:
                    response = ""
            
                lock.acquire()
                output_res.append((response, price))
                progress_bar.update(1)
                lock.release()
            
            thread = threading.Thread(target=_my_function,args=(data, output_res, progress_bar, lock))
            self.thread_pools.append(thread)
            thread.start()
        
        for thread in self.thread_pools:
            thread.join()
            
        self.thread_pools.clear()

        valid_response = 0
        valid_price = 0

        for response, price in output_res:
            if response!="<error>":
                valid_response += 1
                valid_price += price
            
        estimated_per_price = valid_price / valid_response
        estimated_total_price = estimated_per_price * dataset_num

        return estimated_per_price, estimated_total_price

    
    def isRateLimited(self)->bool:
        if "RPM" in self.rateLimit:
            return self._isRateLimited_RPM()
        elif "SPR" in self.rateLimit:
            return self._isRateLimited_SPR()
        else:
            print("please check! Not rate limit found!")
            return False


    def _isRateLimited_RPM(self)->bool:
        cur_time = datetime.now().minute
        if cur_time != self._minute:
            # time diff, reset avail request
            self._avail_req = self.rateLimit["RPM"]
            self._minute = cur_time
        
        if self._avail_req > 0:
            self._avail_req -= 1
            return False
        else:
            return True


    def _isRateLimited_SPR(self)->bool:
        cur_time = datetime.now()
        diff = (cur_time-self._lastTime).total_seconds()
        if diff >= (self.rateLimit["SPR"]+0.1):
            self._lastTime = cur_time
            return False
        else:
            return True        
    

    def dataset_generate(self, input_dataset, output_file, batch_size=10):
        """
        Args:
            input_dataset ([list|Dataset]): the input dataset
            output_file (str): the target save path
            batch_size (int, optional): write frequence. Defaults to 50.
        """
        
        # thread func
        def my_function(index, data, output_res, progress_bar, lock):
            max_req = 5
            req_time = 0
            while req_time<max_req:
                req_time += 1
                try:
                    response = self.generate(data)
                    # print(response)
                    if response!="<error>":
                        break
                except:
                    while True:
                        lock.acquire()
                        rateLimit = self.isRateLimited()
                        lock.release()
                        if rateLimit:
                            time.sleep(1)
                        else:
                            t = random.random() + 1
                            time.sleep(t)
                            break
                        

            if req_time >= max_req:
                print("Intnet Error!")
                response = "<error>"
                

            if response in ["", None]:
                response = ""
            
            lock.acquire()
            output_res.append((index,response))
            progress_bar.update(1)
            lock.release()
        
        
        lock = threading.Lock()
        err_responses = []
        
        if os.path.exists(output_file):
            line = 0
            with open(output_file) as f:
                while True:
                    res = f.readline().strip()
                    if not res:
                        break
                    
                    # res = eval(res)
                    if res == "<error>":
                        try:
                            err_responses.append((line, input_dataset[line]))
                        except:
                            print("output file error!")
                            exit()
                    line+=1
                    
            if line != len(input_dataset):
                print(f"Continue from Line {line}")
            
            start_point = line
        else:
            start_point = 0
        
        dataset_nums = len(input_dataset)
        current_num = start_point
        
        progress_bar = tqdm.tqdm(total=(dataset_nums-start_point))
        
        
        while current_num < dataset_nums:
            start_n = progress_bar.n
            end_num = current_num+batch_size if (current_num+batch_size) <= dataset_nums else dataset_nums
            batch_data = input_dataset[current_num: end_num ]
            current_num += len(batch_data)
            
            output_res = []
            # call api
            for index,data in enumerate(batch_data):    
                while True:
                    lock.acquire()
                    rateLimit = self.isRateLimited()
                    lock.release()
                    if rateLimit:
                        time.sleep(1)
                    else:
                        break  
                thread = threading.Thread(target=my_function,args=(index, data, output_res, progress_bar, lock))
                self.thread_pools.append(thread)
                thread.start()
                try:
                    sleep_time =  1 / self.rateLimit["RPM"]
                except:
                    sleep_time = 0
                time.sleep(sleep_time)
            
            for thread in self.thread_pools:
                thread.join()
                
            
            self.thread_pools.clear()
            
            # sort response
            sorted_responses = sorted(output_res, key=lambda x: x[0])
            responses = [i[1] for i in sorted_responses]
        
            # write result
            with open(output_file, "a") as f:
                for i,res in enumerate(responses):
                    if res == "<error>":
                        # record error responses
                        err_responses.append((current_num+i, batch_data[i]))
                    f.write(repr(res) + "\n")
        
        progress_bar.close()
        
        corrected_responses = []
        turn = 1
        while len(err_responses) > 0:
            turn += 1
            print(f"turn {turn} 重新处理 {len(err_responses)} 个错误请求")
            pbar = tqdm.tqdm(total=len(err_responses))
            output_responses = []
            # call api
            for err_res in err_responses:    
                while True:
                    lock.acquire()
                    rateLimit = self.isRateLimited()
                    lock.release()
                    if rateLimit:
                        time.sleep(1)
                    else:
                        break
    
                thread = threading.Thread(target=my_function,args=(err_res[0], err_res[1], output_responses, pbar, lock))
                self.thread_pools.append(thread)
                thread.start()
            
            for thread in self.thread_pools:
                thread.join()
            
            self.thread_pools.clear()
            
            # sort response
            sorted_responses = sorted(output_responses, key=lambda x: x[0])
            
            still_err_responses = []
            for i in range(len(err_responses)):
                if sorted_responses[i][1] == "<error>":
                    still_err_responses.append(err_responses[i])
                else:
                    corrected_responses.append(sorted_responses[i])
            
            err_responses = still_err_responses

            pbar.close()
            
        if len(err_responses) == 0:
            print("Finish all errors")
        else:
            print(f"There are {len(err_responses)} errors.")
            
        import tempfile
        corrected_indice = [i[0] for i in corrected_responses]
        if corrected_responses != []:
            with open(output_file, "r") as f, tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
                i=0
                while True:
                    line = f.readline().strip()
                    if not line:
                        break
                    if eval(line)!="<error>":
                        tmp_file.write(line+"\n")
                    else:
                        if i in corrected_indice:
                            index = corrected_indice.index(i)
                            corr_response = repr(corrected_responses[index][1])
                            tmp_file.write(corr_response+"\n")
                        else:
                            tmp_file.write("'<error>'\n")
                    i+=1
            shutil.move(tmp_file.name, output_file)
            
            
        
    
    def file_generate(self, input_file, output_file, batch_size=1000):
        """
        在给定的速率下，发起多个请求

        该函数假设输入文件的格式为jsonl, 并且每行都包含prompt字段
        """

        # check if file exist
        import os
        assert os.path.exists(input_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)


        current_pos = 0

        result = subprocess.run(["wc", "-l", input_file], capture_output=True, text=True)
        line_count = int(result.stdout.strip().split()[0])
        progress_bar = tqdm.tqdm(total=line_count)

        lock = threading.Lock()
        
        while current_pos < line_count:
            # get new data
            with open(input_file,"r") as file:
                # 一次读取batch_size行
                batch = []
                for line in itertools.islice(file, current_pos, current_pos+batch_size):
                    batch.append(line)
                current_pos += len(batch)
            try:
                batch_data = [json.loads(line.strip())["prompt"] for line in batch]
            except:
                print("not find prompt key in data file")
                exit()
            
            output_res = []
            # call api
            for index,data in enumerate(batch_data):    
                while True:
                    lock.acquire()
                    rateLimit = self.isRateLimited()
                    lock.release()
                    if rateLimit:
                        time.sleep(1)
                    else:
                        break

                def my_function(index, data, output_res, progress_bar, lock):
                    max_req = 10
                    req_time = 0
                    while req_time<max_req:
                        req_time += 1
                        try:
                            response = self.generate(data)
                            if response!="<error>":
                                break
                        except:
                            while True:
                                lock.acquire()
                                rateLimit = self.isRateLimited()
                                lock.release()
                                if rateLimit:
                                    time.sleep(1)
                                else:
                                    break

                    if req_time >= max_req:
                        print("Intnet error.")
                        response = "<error>"

                    if response in ["", None]:
                        response = ""
                    
                    lock.acquire()
                    output_res.append((index,response))
                    progress_bar.update(1)
                    lock.release()
                
                
                thread = threading.Thread(target=my_function,args=(index, data, output_res, progress_bar, lock))
                self.thread_pools.append(thread)
                thread.start()
            
            # 等待所有线程执行完毕
            for thread in self.thread_pools:
                thread.join()
            
            self.thread_pools.clear()
            
            # sort response
            sorted_responses = sorted(output_res, key=lambda x: x[0])
            responses = [i[1] for i in sorted_responses]
            # write result
            with open(output_file, "a") as f:
                for res in responses:
                    output = {"response": res}
                    output_str = json.dumps(output, ensure_ascii=False)
                    f.write(output_str + "\n")
        
        progress_bar.close()
    

    def recurring_response_get(self, input_file, output_file, batch_size=1000):  

        # check if file exist
        import os
        assert os.path.exists(input_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        rounds=0
        errors=[]
        correction=[]
        lock = threading.Lock()
        output_res=[]
        current_pos = 0
        with open(input_file,"r") as f:
            first_line=f.readline()
        f.close()
        line=json.loads(first_line.strip())
        if not("prompt" in line):
            print("wrong input file!")
            
        if(os.path.exists(output_file)):
            print("already have output_file")
        else:
            self.file_generate(input_file, output_file, batch_size)
        
        import jsonlines
        with open(output_file,'r')as f2:
            for index,data in enumerate(jsonlines.Reader(f2)):
                if(data['response']=='<error>'):
                    line=linecache.getline(input_file, int(index+1))
                    print(input_file,index)
                    print(line)
                    line_data=json.loads(line)
                    errors.append((index,line_data['prompt']))
        output_res=[]    
        while(rounds<=5):
            correction=[]
            time.sleep(5)
            if(len(errors)==0):
                print(f"congratulations! no errors in round {rounds}")
                break
            else:
                print(f"round {rounds}!")
            line_count=len(errors)
            progress_bar = tqdm.tqdm(total=line_count)
            for index,error in enumerate(errors):
                while True:
                    lock.acquire()
                    rateLimit = self.isRateLimited()
                    lock.release()
                    if rateLimit:
                        time.sleep(1)
                    else:
                        break
                def correct_function(error, output_res,correction, progress_bar, lock):
                    max_req = 10
                    req_time = 0
                    while req_time<max_req:
                        req_time += 1
                        try:
                            response = self.generate(error[1])
                            if response!="<error>":
                                break
                        except:
                            # 重新发起请求前确保不会超过速率限制
                            while True:
                                lock.acquire()
                                rateLimit = self.isRateLimited()
                                lock.release()
                                if rateLimit:
                                    time.sleep(1)
                                else:
                                    break
                    if req_time >= max_req:
                        print("多次网络请求错误")
                        response = "<error>"
                    if response in ["", None]:
                        response = ""
                    lock.acquire()
                    if(response!='<error>'):
                        progress_bar.update(1)
                        output_res.append((error[0],response))
                        correction.append(error)
                    lock.release()
                thread = threading.Thread(target=correct_function,args=(error, output_res, correction,progress_bar, lock))
                self.thread_pools.append(thread)
                thread.start()
            for thread in self.thread_pools:
                thread.join()
            self.thread_pools.clear()
            for error in correction:
                if(error in errors):
                    errors.remove(error)
                else:
                    print(error)
                    print(errors)
            progress_bar.close() 
            if(len(errors)!=0):
                num=len(errors)
                print(f"There are {num} errors left in our response,the index of them are:")
                for error in errors:
                    print(error[0])
            rounds+=1
        if(len(errors)!=0):
            for error in errors:
                output_res.append((error))
        if(len(output_res)!=0):
            sorted_responses = sorted(output_res, key=lambda x: x[0])
            responses = [i[1] for i in sorted_responses]
            print(f"the length of sorted_responses is {len(sorted_responses)}")
            for index,data in enumerate(sorted_responses):
                print(index,data)
            import tempfile

            with open(output_file, "r") as f, tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
                i=0
                for index,data in enumerate(jsonlines.Reader(f)):
                    if(data['response']!="<error>"):
                        json_data=json.dumps(data, ensure_ascii=False)
                        tmp_file.write(json_data+ "\n")
                    else:
                        if(index==sorted_responses[i][0]):
                            temp={"response":responses[i]}
                            json_data=json.dumps(temp, ensure_ascii=False)
                            tmp_file.write(json_data+ "\n")
                        else:
                            print(f'wrong in {index}!!!!!')
                        i+=1
            
            shutil.move(tmp_file.name, output_file)

        
class ChatGpt(BaseModel):
    def __init__(self, model="gpt-3.5-turbo",api_key=None) -> None:
        # gpt-3.5-turbo
        # gpt-4
        rateLimit={
            "RPM":200
            }
        
        super().__init__(rateLimit)           
        self.model=model
        self.api_key=api_key
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.configure_params(temperature=0)
    
    from wrapt_timeout_decorator import timeout
    
    @staticmethod
    @timeout(240)
    def completions_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def raw_request(self, model, messages, temperature, timeout=5):
        import requests

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.api_key)
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=data, timeout=timeout) 
        result = response.json()
        return result          
    
    def generate(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        openai.api_base = self.api_base
        response = self.completions_with_backoff(
        model=self.model,
        messages=messages,
        temperature=self.temperature,
        )

        return response.choices[0].message["content"]    

    

api_secret_keys = {
    "ChatGpt" : ""
}

def main(args):
    model_class = globals()[args.model]
    if args.api_secret_key == "default":
        args.api_secret_key = api_secret_keys[args.model]

    model = model_class(api_key=args.api_secret_key)

    if args.estimate:
        percost,totalcost = model.estimate_cost(args.input_file, args.sample_number)
        print(f"For this dataset, estimated per request cost is: ** {percost} ** , estimated total cost is: ** {totalcost} **")
    else:
        # model.recurring_response_get(args.input_file, args.output_file, args.batch_size)
        model.recurring_response_get(args.input_file, args.output_file, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")

    parser.add_argument("--model", type=str, help="Model class name", choices=["ChatGpt"])
    parser.add_argument("--api_secret_key", type=str, help="Your api_secret_key", default="default")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size (default: 1000)")
    parser.add_argument("--input_file", type=str, default="./test.jsonl", help="Input file path (default: ./test.jsonl)")
    parser.add_argument("--output_file", type=str, default="./response.txt", help="Output file path (default: ./response.txt)")

    parser.add_argument("--estimate", action="store_true", help="whether to estimate the price")
    parser.add_argument("--sample_number", type=int, default=100, help="the sampled dataset number used to estimate the api call price.")

    args = parser.parse_args()
    main(args)