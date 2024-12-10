from time import sleep
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdb
import re
import json

header = {
    'user-agent': '',
    'cookie': '',
    # 'origin':'https://www.douban.com',
    # 'Referer':'https://www.douban.com',
}


def get_topic_info(url_list):
    '''
    根据讨论贴url列表，获取讨论贴的基本信息，返回DataFrame。
    url_list: 讨论贴url列表。
    content_df: 讨论贴内容(每贴第一条)的DataFrame。
    comment_df: 讨论贴回复的评论内容的DataFrame。
    '''
    content_df = pd.DataFrame(
        columns=['url', 'user', 'user_url', 'title', 'content', 'time', 'react_num', 'collect_num', 'timestamp'])
    comment_df = pd.DataFrame(columns=[
        'url', 'comment_id', 'user', 'user_url', 'comment', 'time', 'reply_to', 'timestamp'])
    flag = True

    for url in url_list:
        if "topic" in url:
            if "307423713" in url:
                flag = True
            if flag:
                for _ in range(50):
                    try:
                        response = requests.get(url, headers=header)
                        soup = BeautifulSoup(response.text, 'lxml')
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print("url:%s, Time:%s" % (url, timestamp))

                        # 顶楼部分
                        content_raw = soup.find('div', class_='topic-doc')
                        if "有异常请求" in str(soup):
                            raise ValueError
                        if content_raw is None:
                            print('url:%s, content_raw is None' % url)
                            break
                        user = content_raw.select('h3 > .from > a')[0].get_text()
                        user_url = content_raw.select('h3 > .from > a')[0]['href']
                        time = content_raw.find('span', class_='create-time color-green').get_text()
                        script_tag = content_raw.find('script', text=re.compile('window._CONFIG.topic'))
                        match = re.search(r"'title':\s*'([^']+)'", script_tag.string)
                        if match:
                            title = match.group(1)
                        else:
                            title = ""
                        content = content_raw.find(
                            'div', id='link-report').get_text().strip()

                        content_col = {'url': url, 'user': user, 'user_url': user_url, 'title': title, 'content': content,
                                    'time': time, 'timestamp': timestamp}
                        # print(content_col)
                        content_df = pd.concat([content_df, pd.DataFrame([content_col])], ignore_index=True)

                        # 评论部分
                        comment_raw = soup.find('ul', id='comments')
                        if comment_raw is None:
                            print('url:%s, comment_raw is None' % url)
                            break
                        
                        comment_list = comment_raw.find_all('li')
                        if comment_list is None or len(comment_list) == 0:
                            print('url:%s, comment_list is Empty' % url)
                            break
                        
                        page = 1

                        # 获取翻页按钮
                        while len(soup.select('.paginator > .next > a')) > 0:
                            com_res = requests.get(
                                url, params={'start': page*100}, headers=header, verify=False)
                            page = page+1
                            soup = BeautifulSoup(com_res.text, 'lxml')
                            comment_raw = soup.find('ul', id='comments')
                            # 将下一页的评论列表追加进common_list
                            comment_list = comment_list + comment_raw.find_all('li')
                            sleep(5)

                        for com in comment_list:
                            # print(com)
                            comment_id = com.get('data-cid')
                            user = com.select('.reply-doc > div > h4 > a ')[0].get_text()
                            user_url = com.select('.reply-doc > div > h4 > a ')[0]['href']
                            time = com.select(
                                '.reply-doc > div > h4 > .pubtime')[0].get_text()
                            comment = com.find('p')
                            if comment is not None:
                                comment = comment.get_text().strip()
                            if com.find('div', class_='reply-quote-content') is None:
                                reply_to = ''
                            else:
                                reply_to = com.find(
                                    'div', class_='reply-quote-content').get('data-ref-cid')
                            comment_col = {'url': url, 'comment_id': comment_id, 'user': user, 'user_url': user_url,
                                        'time': time, 'comment': comment, 'reply_to': reply_to, 'timestamp': timestamp}
                            # print(comment_col)
                            comment_df = pd.concat([comment_df, pd.DataFrame([comment_col])], ignore_index=True)
                        break
                    except Exception as e:
                        print(e)
                        print('Error in ', url)
                        sleep(30)
                sleep(20)
                content_df.to_csv('discussion_content.csv',
                            index=False, encoding='utf-8-sig')
                comment_df.to_csv('discussion_reply.csv',
                                index=False, encoding='utf-8-sig')
        elif "people" in url:
            if "307423713" in url:
                flag = True
            if flag:
                for _ in range(50):
                    try:
                        response = requests.get(url, headers=header)
                        sleep(30)
                        soup = BeautifulSoup(response.text, 'lxml')
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print("url:%s, Time:%s" % (url, timestamp))

                        content_raw = soup.find("div", class_='status-item')
                        if content_raw is None:
                            print('url:%s, content_raw is None' % url)
                            break
                        if content_raw is not None:
                            content = content_raw.find('div', class_='status-title')
                            if content is not None:
                                title = content.get_text(strip=True)
                            else:
                                title = ""

                            content = content_raw.find("p")
                            if content is not None:
                                content = content.get_text(strip=True)
                            else:
                                content = ""

                            meta_data_raw = content_raw.find('div', class_='text')
                            try:
                                user_url = meta_data_raw.find('a', class_='lnk-people')['href']
                                user = meta_data_raw.find('a', class_='lnk-people').get_text(strip=True)
                                time = meta_data_raw.find('span', class_='pubtime').get_text(strip=True)
                            except Exception as e:
                                print(e)
                                print('url:%s, something is None' % url)
                                break

                            content_col = {'url': url, 'user': user, 'user_url': user_url, 'title': title, 'content': content,
                                    'time': time, 'timestamp': timestamp}
                            content_df = pd.concat([content_df, pd.DataFrame([content_col])], ignore_index=True)

                        comment_raw = soup.find('script', text=re.compile(r'var _COMMENTS_CONFIG'))
                        if comment_raw is not None:
                            num = 0
                            while True:
                                comment_all_list = comment_raw.string
                                comment_begin = comment_all_list.find("'comments':")
                                comment_end = comment_all_list.find("'total':")
                                total_end = comment_all_list.find("'start'")

                                comment_all = comment_all_list[comment_begin: comment_end]
                                total_num = comment_all_list[comment_end: total_end]
                                total_num = int(re.search(r"'total': (\d+)", total_num).group(1))

                                comment_list = json.loads(comment_all[12:-7])
                                for comments in comment_list:
                                    comment_id = comments["id"]
                                    user = comments["author"]["name"]
                                    user_url = comments["author"]["url"]
                                    time = comments["create_time"]
                                    comment = comments["text"]
                                    reply_to = "" if comments["parent_comment_id"] == '0' else comments["parent_comment_id"]
                                    comment_col = {'url': url, 'comment_id': comment_id, 'user': user, 'user_url': user_url,
                                            'time': time, 'comment': comment, 'reply_to': reply_to, 'timestamp': timestamp}
                                    
                                    # print(comment_col)
                                    comment_df = pd.concat([comment_df, pd.DataFrame([comment_col])], ignore_index=True)

                                    if len(comments['replies']) != 0:
                                        for replies in comments['replies']:
                                            comment_id = replies["id"]
                                            user = replies["author"]["name"]
                                            user_url = replies["author"]["url"]
                                            time = replies["create_time"]
                                            comment = replies["text"]
                                            reply_to = replies['ref_comment']['id']
                                            comment_col = {'url': url, 'comment_id': comment_id, 'user': user, 'user_url': user_url,
                                                    'time': time, 'comment': comment, 'reply_to': reply_to, 'timestamp': timestamp}
                                            
                                            # print(comment_col)
                                            comment_df = pd.concat([comment_df, pd.DataFrame([comment_col])], ignore_index=True)
                                    
                                num += 20
                                if num >= total_num:
                                    break
                                else:
                                    response = requests.get(url + f"/?start={num}", headers=header)
                                    sleep(30)
                                    soup = BeautifulSoup(response.text, 'lxml')
                                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    print("url:%s, Time:%s" % (url, timestamp))

                                    comment_raw = soup.find('script', text=re.compile(r'var _COMMENTS_CONFIG'))
                                    if comment_raw is None:
                                        break
                        content_df.to_csv('discussion_content.csv',
                                    index=False, encoding='utf-8-sig')
                        comment_df.to_csv('discussion_reply.csv',
                                    index=False, encoding='utf-8-sig')
                        break
                    except Exception as e:
                        print(e)
                        sleep(30)
        
    return content_df, comment_df

if __name__ == '__main__':

    with open("url_姜萍.json", "r") as f:
        url_list = json.load(f)
    with open("url_数学竞赛中专.json", "r") as f:
        url_list += json.load(f)    
    content_df, comment_df = get_topic_info(url_list)
    content_df.to_csv('discussion_content.csv',
                      index=False, encoding='utf-8-sig')
    comment_df.to_csv('discussion_reply.csv',
                    index=False, encoding='utf-8-sig')