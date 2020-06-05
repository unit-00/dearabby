import requests 
from bs4 import BeautifulSoup
import time

class Scraper:
    def __init__(self, archives_coll, articles_coll, year_start=1991, year_end=2021):
#         Expect two mongo collections, one for archives and one for articles
        self.archives_coll = archives_coll
        self.articles_coll = articles_coll
        self.year_start = year_start
        self.year_end = year_end
        self.base_url = 'https://www.uexpress.com/dearabby'
    
    def scrape(self):
#         Scrape and insert into warehouse, return lists of archives and status code
        base_url = self.base_url
        archive_status = []
        article_status = []

        for i in range(self.year_start, self.year_end):
            print('year:', i)
            print()
            url = base_url + '/archives/' + str(i)

            status_code, archive = self.scrape_insert(self.archives_coll, url)

            archive_status.append({url: status_code})

            if status_code == 200:
                status_codes = self.scrape_articles(self.articles_coll, archive)

                article_status.append({url: status_codes})

            if (i - year_start) == 5:
                print('sleep 30 in archive')
                print(url)
                print()
                time.sleep(30)

            if (i - year_start) == 10:
                print('sleep 60 in archive')
                print(url)
                print()
                time.sleep(60)


        return archive_status, article_status

    def scrape_articles(self, coll, archive):
        soup = BeautifulSoup(archive, 'lxml')
        base_url = 'https://www.uexpress.com'

        list_of_articles = soup.select('article.media.media-item.list-item--large')

        status = []

        for i, article in enumerate(list_of_articles):
            url = base_url + article.select('a.media__link--primary')[0]['href']

            status_code, article = self.scrape_insert(coll, url)

            status.append({'url': url,
                           'status_code': status_code, 
                           'article': article})

            if i % 10 == 9:
                print('sleep 30 in article')
                print(url)
                print()
                time.sleep(30)

        return status



    def scrape_insert(self, coll, url):
        response = requests.get(url)

        print('sleep 2 seconds')
        print(url)
        print()
        time.sleep(2)

        coll.insert_one({'url': url,
                         'status_code': response.status_code,
                         'html': response.content})

        return response.status_code, response.content
    
    def clean_articles(self, cleaned_coll):
        for article in self.articles_coll.find({}, {'_id': 0, 'url': 1, 'html': 1}):
            
            url, html = article.keys()

            soup = BeautifulSoup(article[html])

            qa = self.extract_qa(soup)

            cleaned_coll.insert_one({'url': url,
                                     'extracted_qa': qa})
            
    def extract_qa(self, soup):
        articles = soup.select('article.item-section')

        year, month, day = soup.select('time')[0]['datetime'].split('-')
        titles = soup.select('h1.item-title')
        qa = []

        for i, a in enumerate(articles):
            q_idx = 0
            a_idx = 0

            title = titles[i] if i < len(titles) else titles[0]

            question = []
            answer = []

            for j, p in enumerate(a.select('p')):
                if p.text.startswith('DEAR'):
                    if p.text.startswith('DEAR ABBY:'):
                        q_idx = j
                        is_q = True
                        question.append(p.text)

                    else:
                        a_idx = j
                        is_q = False
                        answer.append(p.text)

                elif is_q:
                    question.append(p.text)

                else:
                    answer.append(p.text)

            cat = soup.select('article.item-section span')

            qa.append({'year': year,
                       'month': month,
                       'day': day,
                       'title': title.text.strip(),
                       'question': ' '.join(question),
                       'answer': ' '.join(answer),
                       'letterid': i,
                       'categories': cat[i].text if i < len(cat) else []
                      })

            return qa
        
            