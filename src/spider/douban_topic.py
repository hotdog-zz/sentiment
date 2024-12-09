import random
import pandas as pd
import requests
from time import sleep
import pandas
import json
import random
import os
import threading
import pdb
import time

# 请求头
headers = {
    'cookie': '',
    'origin': 'https://www.douban.com',
    'referer': 'https://www.douban.com/search',
    'user-agent': ''
}
url_global = 'https://m.douban.com/rexxar/api/v2/search'

def get_url(query):
    start = 0
    max_num = 20
    while True:
        for i in range(50):
            try:
                post_data = {'q': query, 'type': '' , 'loc_id': '', 'start': start, 'count': '10', 'sort': 'relevance', 'ck': '_C4C'}
                json_text = requests.post(url=url_global, headers=headers, data=post_data).json()
                max_num = json_text['contents']['total']
                items = json_text['contents']['items']

                for item in items:
                    try:
                        create_time = item['target']['create_time']
                        target_id = item['target_id']
                        uri = item['target']['owner']['uri']
                        if 'group' in uri:
                            url = "https://www.douban.com/group/topic/" + str(target_id)
                        elif 'user' in uri:
                            url = item['target']['owner']['url'] + "status/" + str(target_id)
                        else:
                            url = item
                        with open("url.json", "a+") as f:
                            f.write(json.dumps({"url": url, "create_time": create_time}))
                            f.write("\n")
                    except Exception as e:
                        pass
                break
            except Exception as e:
                print("Wrong")
                print(i)
                time.sleep(30)
        print(start)
        start += 10
        if start > max_num:
            break
        time.sleep(15)

for query in ['数学竞赛中专']:
    get_url(query)
# '姜萍', '数学竞赛中专'