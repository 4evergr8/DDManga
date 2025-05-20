import requests
from lxml import html
import os


authors = [
    "kedama-gyuunyuu",
    "ponkichi",
    "koniro-club",
    "deadflow",
    "karory",
    "kenja-time",
    "orangemaru",
    "mignon",
    "torimogura",
    "blvefo9"
]


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
base_url = 'https://nhentai.net'

for author in authors:
    print(f"开始处理作者：{author}")

    # 构造搜索 URL
    search_url = f'{base_url}/search/?q={author}+fullcolor+japanese'

    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"访问失败：{search_url}")
            continue

        tree = html.fromstring(resp.content)
        hrefs = tree.xpath('//div[@class="gallery"]/a/@href')
        full_urls = [base_url + href for href in hrefs]

        # 创建对应作者文件夹
        save_dir = os.path.join('train', author)
        os.makedirs(save_dir, exist_ok=True)
        print(full_urls)

        for book_url in full_urls:
            for page in range(3, 100):  # 最多尝试到第99页
                page_url = f"{book_url}{page}/"
                book_id = book_url.split('/g/')[1].strip('/')
                try:
                    r = requests.get(page_url, headers=headers, timeout=10)
                    if r.status_code == 404:
                        break

                    tree = html.fromstring(r.content)
                    img_url = tree.xpath('//*[@id="image-container"]/a/img/@src')
                    if not img_url:
                        continue

                    img_url = img_url[0]
                    ext = os.path.splitext(img_url)[-1].split('?')[0]
                    filename = os.path.join(save_dir, f"{book_id}-{page}{ext}")

                    # 下载图片
                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    with open(filename, 'wb') as f:
                        f.write(img_data)

                    print(f"已下载: {filename}")
                    #time.sleep(0.05)

                except Exception as e:
                    print(f"下载失败: {e}")
                    continue

    except Exception as e:
        print(f"作者 {author} 页面请求失败：{e}")

print("全部作者处理完成。")
