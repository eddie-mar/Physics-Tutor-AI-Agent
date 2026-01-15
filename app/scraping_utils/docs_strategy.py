from abc import ABC, abstractmethod
from langchain_core.documents import Document

class AbsStrategy(ABC):
    @abstractmethod
    def get_docs(self, main, chapter_title):
        pass

class Content(AbsStrategy): 
    def get_docs(self, main, chapter_title):
        current_section = 'Introduction'
        buffer = []
        documents = []

        for ele in main.find_all(['p', 'h2'], recursive=True):
            if ele.name == 'p':
                text = ele.get_text(' ', strip=True)
                if text:
                    buffer.append(text)
            elif ele.name == 'h2':                  # <h2> means it is a subtitle, another section
                if buffer:                          # save previous section
                    documents.append(
                        Document(
                            page_content='\n'.join(buffer),
                            metadata={
                                'source': 'OpenStax College Physics 2e',
                                'chapter': chapter_title,
                                'type': 'content',
                                'section': current_section,
                                'license': 'CC BY'
                            }
                        )
                    )
                    buffer = []                     # reset buffer after saved
                current_section = ele.get_text(strip=True)

        if buffer:
            documents.append(
                Document(
                    page_content='\n'.join(buffer),
                    metadata={
                        'source': 'OpenStax College Physics 2e',
                        'chapter': chapter_title,
                        'type': 'content',
                        'section': current_section,
                        'license': 'CC BY'
                    }
                )
            )

        return documents


class Glossary(AbsStrategy):
    def get_docs(self, main, chapter_title):
        documents = []

        for ele in main.find_all('dl'):                         # each key-value pair is an entry
            term = ele.find('dt').text.strip()
            definition = ele.find('dd').text.strip()

            documents.append(
                Document(
                    page_content=f'{term}: {definition}',
                    metadata={
                        'source': 'OpenStax College Physics 2e',
                        'chapter': chapter_title,
                        'type': 'glossary',
                        'term': term,
                        'license': 'CC BY'
                    }
                )
            )
        
        return documents


class SectionSummary(AbsStrategy):
    def get_docs(self, main, chapter_title):
        documents = []

        for ele in main.find_all('section', class_='section-summary'):
            subsection = ele.find('h2').get_text(separator=' ', strip=True)
            h2 = ele.find('h2')
            h2.decompose()                                          # remove subsection in ele since ele content structure is not consistent
            text = ele.get_text(separator='\n', strip=True)         # ele.find('ul').get_text(separator='\n', strip=True) is more general but there are cases where <ul> is not its structure

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        'source': 'OpenStax College Physics 2e',
                        'chapter': chapter_title,
                        'type': 'section summary',
                        'subsection': subsection,
                        'licence': 'CC BY',
                    }
                )
            )
        
        return documents