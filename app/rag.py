from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import scrape_openstax  

load_dotenv()
VECTORSTORE = os.getenv('VECTORSTORE', 'vectorstore')
    
def build_vectorstore_from_openstax(embeddings):
    ''' This function will scrape the document and create vectorstore '''
    documents = scrape_openstax.scrape_book()

    if not documents:
        raise RuntimeError('No documents scraped from OpenStax')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE)

    return vectorstore
