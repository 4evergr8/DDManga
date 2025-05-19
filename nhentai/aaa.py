import requests
from lxml import html
import time
import os


authors = [
    "kedama-gyuunyuu",
    "ponkichi",
    "koniro-club",
    "deadflow",
    "karory"
]

for author in authors:
    print(author)




    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    search_url = f'https://nhentai.net/search/?q={author}+fullcolor+japanese'
    base_url = 'https://nhentai.net'

    # 请求搜索页
    resp = requests.get(search_url, headers=headers)
    tree = html.fromstring(resp.content)

    # 提取每本本子的链接
    hrefs = tree.xpath('//div[@class="gallery"]/a/@href')
    full_urls = [base_url + href for href in hrefs]

    # 创建保存目录
    if not os.path.exists('downloaded'):
        os.mkdir('downloaded')

    # 遍历所有链接
    for book_url in full_urls:
        for page in range(3, 100):  # 最多尝试到第99页
            page_url = f"{book_url}{page}/"
            try:
                r = requests.get(page_url, headers=headers)
                if r.status_code != 200:
                    break
                tree = html.fromstring(r.content)
                img_url = tree.xpath('//*[@id="image-container"]/a/img/@src')
                if not img_url:
                    continue
                img_url = img_url[0]

                # 提取扩展名
                ext = os.path.splitext(img_url)[-1].split('?')[0]
                timestamp = int(time.time() * 1000)
                filename = f"downloaded/{timestamp}{ext}"

                # 下载图片
                img_data = requests.get(img_url, headers=headers).content
                with open(filename, 'wb') as f:
                    f.write(img_data)
                print(f"已下载: {filename}")
                time.sleep(0.05)
            except Exception as e:
                print(f"错误: {e}")
                continue
