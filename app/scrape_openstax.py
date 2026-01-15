import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from bs4 import BeautifulSoup
import requests

from logger import get_logger
from scraping_utils import get_soup, remove_unnecessary, get_next_link, get_documents

logger = get_logger(name=__name__, log_file='scraper.log')

load_dotenv()
START_URL = os.getenv('START_URL', 'https://openstax.org/books/college-physics-2e/pages/1-introduction-to-science-and-the-realm-of-physics-physical-quantities-and-units')
END_URL = os.getenv('END_URL', 'https://openstax.org/books/college-physics-2e/pages/34-section-summary')

def scrape_url(url: str):
    ''' This function is used to scrape per page of the book '''

    try:
        soup = get_soup(url)                            # get parsed html
    except Exception:
        raise

    soup = remove_unnecessary(soup)

    try:
        if url == END_URL:     # scraping endpoint for the book
            next_link = None
        else:
            next_link = get_next_link(soup)
    except RuntimeError:
        raise
    except Exception:
        raise

    chapter_title = soup.find('title')
    chapter_title = chapter_title.get_text() if chapter_title else 'Unknown Chapter'

    main = soup.find('main')
    if not main:                                        # if main is not found, just return next_link so we can continue scraping
        return {'docs': None, 'href': next_link}
    
    documents = get_documents(main, chapter_title)

    return {'docs': documents, 'href': next_link}
    

def scrape_book(start_url: str = START_URL) -> list[Document]:
    ''' This function is used to scrape the whole book using the function scrape_url() '''

    logger.info(f'STARTING SCRAPER at {start_url}')

    all_docs = []
    url = start_url

    retries = 3

    while url:
        if retries <= 0:
            logger.error(f'Maximum retries reached for {url}. Exiting scraper.')
            break
        
        try: 
            section_content = scrape_url(url)       # returns document and next link
            if section_content['docs']:             # might return empty docs if site <main> is nonexistent
                all_docs.extend(section_content['docs'])
            url = section_content['href']           # get next page in the book to scrape

            logger.info(f'Scraped {url} successfully.')
            retries = 3                             # reset retries for successful scrape
        except requests.exceptions.TimeOut:
            retries -= 1 
            logger.error(f'Timeout while scraping {url}')
        except requests.exceptions.RequestException as e:
            retries -= 1
            logger.error(f'Request failed for {url}: {e}')
        except requests.exceptions.HTTPError as e:
            retries -= 1
            logger.error(f'HTTPError for {url}: {e}')
        except RuntimeError as e:
            logger.error(f'Runtime error occured for {url}. Next link cannot be found.')
            break
        except Exception as e:
            retries -= 1
            logger.exception(f'Unexpected error for {url}: {e}')

    logger.info(f'ENDED SCRAPING in {url}')
    return all_docs


        

    

