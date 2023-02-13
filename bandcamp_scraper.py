import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from pathlib import Path

options = webdriver.ChromeOptions()
options.add_argument("--headless")

# Create a new instance of the Chrome driver
driver = webdriver.Chrome(options=options)

num_pages = 50
dfs = []

url = "https://bandcamp.com/"
genre_tags = [
    'electronic',
    'rock',
    'metal',
    'alternative',
    'hip-hop-rap',
    'experimental',
    'punk',
    'folk',
    'pop',
    'ambient'
]

for genre_tag in tqdm(genre_tags):

    img_urls = []
    genres_text = []
    titles_text = []
    artists_text = []

    print(genre_tag)
    url = f"https://bandcamp.com/?g={genre_tag}&s=top&p=0&gn=0&f=all&w=0"
    download_path = Path(f'album_covers/{genre_tag}')
    download_path.mkdir(parents=True, exist_ok=True)

    driver.get(url)
    for page in tqdm(range(num_pages)):
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        img_tags = soup.find_all('img', attrs={'class': 'art'})
        img_urls += [img['src'] for img in img_tags]
        genres = soup.find_all('span', attrs={'class': 'item-genre'})
        genres_text += [genre.text for genre in genres]
        titles = soup.find_all('a', attrs={'class': 'item-title'})
        titles_text += [title.text for title in titles]
        artists = soup.find_all('a', attrs={'class': 'item-artist'})
        artists_text += [artist.text for artist in artists]

        next_button = driver.find_elements(By.CLASS_NAME, "item-page")[-1]
        assert(next_button.text == 'next')
        next_button.click()

    print('downloading..')
    file_names = [url.split("/")[-1] for url in img_urls]
    df = pd.DataFrame({
        'genre': genres_text,
        'artist': artists_text,
        'title': titles_text,
        'filename': file_names,
        'url': img_urls
    })

    df = df.drop_duplicates()
    print(f'downloading {len(df)} files..')
    for url, filename in tqdm(zip(df.url, df.filename)):
        url = url[:-5] + "2" + url[-4:]
        urllib.request.urlretrieve(url, Path(download_path, filename))
    dfs.append(df)

data_df = pd.concat(dfs).drop_duplicates()
data_df.to_csv('data.csv', index=False)

driver.quit()