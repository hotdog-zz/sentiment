from time import sleep
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdb
import re
import json
from copy import deepcopy

headers = {
    'user-agent': '',
    'cookie': '',
}

def get_content(url, comment_df):

    def get_page_content(comment_df, url):

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(20):
            try:
                response = requests.get(url=url, headers=headers)
                break
            except Exception as e:
                print(e)
                print("Wrong " + url)
                sleep(30)
        soup = BeautifulSoup(response.text, 'lxml')

        page_num_raw = soup.find('div', class_='p_thread thread_theme_5')
        if page_num_raw is None and "404页面" in str(soup):
            return comment_df, 0, 0, "404页面"
        page_num = page_num_raw.find('li', class_='l_reply_num').find_all('span', class_='red')[-1].get_text(strip=True)
        page_num = int(page_num)

        content_raw = soup.find('div', class_='p_postlist')
        content_lists = content_raw.find_all('div', class_='l_post')

        for idx, content_list in enumerate(content_lists):
            forum_raw = content_list.get('data-field')
            comment_id = content_list.get('data-pid')

            match = re.search(r'"forum_id":(\d+)', forum_raw)
            # 如果匹配成功，输出 forum_id
            if match:
                forum_id = match.group(1)

            match = re.search(r'"user_id":(\d+)', forum_raw)
            # 如果匹配成功，输出 forum_id
            if match:
                user_id = match.group(1)
            else:
                user_id = ""

            match = re.search(r'"user_name":"(.*?)"', forum_raw)
            if match:
                user_name = match.group(1)
            else:
                user_name = ""

            content = content_list.find('div', class_='d_post_content')
            text = content.get_text(strip=True)

            time_raw = content_list.find('div', class_='post-tail-wrap')
            if time_raw is None:
                match = re.search(r'"date":"(.*?)"', forum_raw)
                if match:
                    text_time = match.group(1)
            else:
                time_raw = time_raw.find_all('span')
                for span in reversed(time_raw):
                    text_time = span.get_text(strip=True)
                    if '2024-' in str(text_time) or '2023-' in str(text_time):
                        break

            comment_row = {
                'url': url, 
                'comment_id': comment_id, 
                'post_id': "", 
                'user_name': user_name, 
                'user_id': user_id, 
                'comment': text, 
                'time': text_time, 
                'my_timestamp': timestamp}
            if idx == 0:
                return_row = deepcopy(comment_row)
            comment_row_df = pd.DataFrame([comment_row])
            comment_df = pd.concat([comment_df, comment_row_df], ignore_index=True)

        comment_df.to_csv('tiebareverse_reply.csv', index=False, encoding='utf-8-sig')
        try:
            # 尝试访问变量
            forum_id
        except NameError:
            print("Wrong forum_id")
            forum_id = "Error"
        return comment_df, forum_id, page_num, return_row
    
    def get_all_content(comment_df, url, post_data):

        timestamp = datetime.now().timestamp()
        timestamp = str(int(timestamp * 1000))
        post_data['t'] = timestamp

        for _ in range(20):
            try:
                response = requests.get(url="https://tieba.baidu.com/p/totalComment", headers=headers, params=post_data).json()
                break
            except Exception as e:
                print(e)
                print("Wrong " + "https://tieba.baidu.com/p/totalComment")
                sleep(30)
        comment_list= response['data']['comment_list']

        if len(comment_list) > 0:
            for key, value in comment_list.items(): 
                comment_info = value['comment_info']
                for comment in comment_info:
                    post_id = comment['post_id']
                    user_name = comment['username']
                    user_id = comment['user_id']
                    text = comment['content']
                    soup_text = BeautifulSoup(text, 'html.parser')
                    text = soup_text.get_text(strip=True)
                    comment_id = comment['comment_id']
                    text_time = datetime.fromtimestamp(comment['now_time'])
                    comment_row = {
                        'url': url, 
                        'comment_id': comment_id, 
                        'post_id': post_id, 
                        'user_name': user_name, 
                        'user_id': user_id, 
                        'comment': text, 
                        'time': text_time, 
                        'my_timestamp': timestamp}
                    comment_row_df = pd.DataFrame([comment_row])
                    comment_df = pd.concat([comment_df, comment_row_df], ignore_index=True)
            comment_df.to_csv('tiebareverse_reply.csv', index=False, encoding='utf-8-sig')
        return comment_df
    
    pn = 1
    while True:      
        url_page = url + f"?pn={pn}"
        tid = url[url.rfind("/")+1:]
        if pn == 1:
            comment_df, forum_id, page_num, return_row = get_page_content(comment_df, url_page)
            if return_row == "Wrong":
                return comment_df, "404页面"
        else:
            comment_df, forum_id, page_num, _ = get_page_content(comment_df, url_page)
        sleep(15)
        if forum_id != 'Error':
            timestamp = datetime.now().timestamp()
            timestamp = str(int(timestamp * 1000))
            post_data = {"t": timestamp,
                        "tid": tid,
                        "fid": str(forum_id),
                        "pn": str(pn),
                        "see_lz": str(0)}
            comment_df = get_all_content(comment_df, url_page, post_data)
            sleep(15)
        pn += 1
        if pn > page_num:
            break
    return comment_df, return_row


if __name__ == '__main__':
    with open("url_贴吧姜萍.json", "r") as f:
        all_url = json.load(f)

    title_df = pd.DataFrame(columns=[
        'url', 'comment_id', 'user_name', 'user_id', 'title', 'time', 'my_timestamp'])
    comment_df = pd.DataFrame(columns=[
        'url', 'comment_id', 'post_id', 'user_name', 'user_id', 'comment', 'time', 'my_timestamp'])
    flag = False
    for url in all_url:
        print(url["url"])
        if "9308866346" in url["url"]:
            flag = True
        if flag:
            comment_df, return_row = get_content(url["url"], comment_df)
            title_row = {
                'url': url["url"], 
                'comment_id': return_row['comment_id'], 
                'user_name': return_row['user_name'], 
                'user_id': return_row['user_id'], 
                'title': url["content"], 
                'time': return_row['time'], 
                'my_timestamp': return_row['my_timestamp']}
            title_row = pd.DataFrame([title_row])
            title_df = pd.concat([title_df, title_row], ignore_index=True)
            title_df.to_csv('tieba_title.csv', index=False, encoding='utf-8-sig')
