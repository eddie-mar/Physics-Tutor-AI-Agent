import re
import os
from dotenv import load_dotenv

from bs4 import BeautifulSoup
import requests

from scraping_utils.docs_strategy import Content, Glossary, SectionSummary

load_dotenv()
BASE_URL = os.getenv('BASE_URL', 'https://openstax.org/books/college-physics-2e/pages/')

# SCRAPING UTILITIES 
def get_soup(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    response.encoding = response.apparent_encoding      # to remove encoding issues in openstax site
    return BeautifulSoup(response.text, 'html.parser')


def remove_unnecessary(soup):
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    outline = soup.find('div', class_='os-chapter-outline')
    objectives = soup.find('section', class_='learning-objectives')
    for part in [outline, objectives]:
        if part:
            part.decompose()

    return soup


def get_next_link(soup, depth=0):
    if depth > 5:
        raise RuntimeError('Too many redirects while finding next link')
    
    unwanted = [r'^\d{1,2}-conceptual-questions', r'^\d{1,2}-problems-exercises']

    next_link = soup.find('a', attrs={'aria-label': 'Next Page'})       # contains link for next page
    if not next_link:
        raise RuntimeError
    
    next_link = next_link['href']

    # we do not include conceptual-questions and problems-exercises
    if re.search(unwanted[0], next_link) or re.search(unwanted[1], next_link):     
        return get_next_link(get_soup(BASE_URL + next_link), depth=depth + 1)            # recursive call thru the unwanted pages until a link is found

    return BASE_URL + next_link                                         # final link


def transform(chapter_title):
    if re.search(r'^Ch\. \d{1,2} Glossary', chapter_title):
        strat = 'glossary'
    elif re.search(r'Ch\. \d{1,2} Section Summary', chapter_title):
        strat = 'section_summary'
    else:
        strat = 'content'

    return strat


def get_documents(main, chapter_title):
    strat = transform(chapter_title)

    mapping = {
        'content': Content,
        'glossary': Glossary,
        'section_summary': SectionSummary
    }

    return mapping[strat]().get_docs(main, chapter_title)





