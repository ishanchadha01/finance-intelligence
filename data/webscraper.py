from bs4 import BeautifulSoup
import requests
import datetime
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def load_stock_data(stock_name, scraped_out):
    
    # Get stock-related news from Reuters as well as date
    df = pd.DataFrame(columns=['Date', 'Text'])
    date = datetime.datetime(2021, 4, 28)
    driver = webdriver.Chrome()
    driver.get('https://www.reuters.com/companies/{}/news'.format(stock_name))

    # Scroll 30 times and get date/description
    scrolls = 20
    links = []
    descriptions = []
    for scroll in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = soup.find_all('div', {'class': 'FeedScroll-feed-container-106s7'})
    for article in articles:
        descriptions.extend(article.find_all('p'))
        links.extend(article.find_all('a'))
    descriptions = [d.string for d in descriptions]
    links = [str(l).split("\"")[3] for l in links]
    for link, description in zip(links, descriptions):
        driver.get(link)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Try to find date information if available
        try:
            dates = soup.find_all('span', {'class': 'DateLine__date___12trWy'})
            datetimes = [date for date in dates]
            if len(datetimes)==0:
                datetimes = soup.find_all('time')
                datetimes = [date for date in datetimes]
            date = datetime.datetime.strptime(datetimes[0].string, '%B %d, %Y')
            df = df.append({'Date': date, 'Text': description}, ignore_index=True)
            df.to_csv(scraped_out, index=False)
            print(df)
            driver.back()
            for scroll in range(scrolls):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)
        except:
            continue



if __name__=='__main__':
    load_stock_data('GE', './raw_data/news_scraped_data.csv')


