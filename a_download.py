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
    "kenja-time",
    "orangemaru",
    "mignon",
    "torimogura",
    "blvefo9",
    "monsieur"
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
base_url = 'https://nhentai.net'

def get_gallery_urls(author):
    print(f"正在获取作者 {author} 的本子链接...")
    search_url = f'{base_url}/search/?q={author}+fullcolor+japanese'
    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"访问失败：{search_url}")
            return []

        tree = html.fromstring(resp.content)
        hrefs = tree.xpath('//div[@class="gallery"]/a/@href')
        full_urls = [(author, base_url + href) for href in hrefs]
        return full_urls
    except Exception as e:
        print(f"获取本子链接失败（作者：{author}）：{e}")
        return []

def download_gallery(author, book_url):
    save_dir = os.path.join('train', author)
    os.makedirs(save_dir, exist_ok=True)
    print(f"开始下载：{book_url}")

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

def main():
    gallery_tasks = []
    for author in authors:
        gallery_tasks.extend(get_gallery_urls(author))

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_gallery, author, url) for author, url in gallery_tasks]
        for future in as_completed(futures):
            future.result()

    print("全部本子下载完成。")

if __name__ == "__main__":
    main()
