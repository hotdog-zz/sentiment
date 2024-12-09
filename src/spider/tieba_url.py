from time import sleep
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdb
import re
import json

headers = {
    'user-agent': '',
    'cookie': '',
}

for i in range(1, 40):
    for _ in range(20):
        print(i)
        try:
            url = f"https://tieba.baidu.com/f/search/res?isnew=1&kw=%CB%EF%D0%A6%B4%A8&qw=jumping&rn=20&un=&only_thread=0&sm=0&sd=&ed=&pn={i}"

            response = requests.get(url=url, headers=headers,)
            soup = BeautifulSoup(response.text, 'lxml')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            url_raw = soup.find('div', class_='s_post_list')
            url_lists = url_raw.find_all('div', class_='s_post')
            for post in url_lists:
                try:
                    # 提取 href
                    url = "https://tieba.baidu.com" + post.find('a', class_='bluelink')['href']
                    
                    # 提取 content
                    content = post.find('span', class_='p_title').text.strip()
                    
                    # 提取 time
                    time = post.find('font', class_='p_green p_date').text.strip()
                    with open("url.json", "a+") as f:
                        f.write(json.dumps({"url": url, "content": content, "time": time}))
                        f.write("\n")
                except Exception as e:
                    continue
            break
        except Exception as e:
            print(e)
            print("Wrong: " + url)
            sleep(30)
    sleep(15)