import requests
from lxml import html
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

authors = [
    "kedama-gyuunyuu",
    "ponkichi",
    "koniro-club",
    "deadflow",
    "karory",
    "sigma",
    "kenja-time",
    "orangemaru",
    "mignon",
    "torimogura",
    "sune",
    "blvefo9"
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
base_url = 'https://nhentai.net'

def process_author(author):
    print(f"开始处理作者：{author}")

    search_url = f'{base_url}/search/?q={author}+fullcolor+japanese'

    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"访问失败：{search_url}")
            return

        tree = html.fromstring(resp.content)
        hrefs = tree.xpath('//div[@class="gallery"]/a/@href')
        full_urls = [base_url + href for href in hrefs]

        save_dir = os.path.join('train', author)
        os.makedirs(save_dir, exist_ok=True)
        print(full_urls)

        for book_url in full_urls:
            for page in range(3, 100):
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

                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    with open(filename, 'wb') as f:
                        f.write(img_data)

                    print(f"已下载: {filename}")

                except Exception as e:
                    print(f"下载失败: {e}")
                    continue

    except Exception as e:
        print(f"作者 {author} 页面请求失败：{e}")

def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_author, author) for author in authors]
        for future in as_completed(futures):
            # 等待任务完成，也可以这里打印状态或捕获异常
            future.result()

if __name__ == "__main__":
    main()
    print("全部作者处理完成。")
