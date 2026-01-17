import os
import time

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from model import llm, embeddings
from rag import build_vectorstore_from_openstax
from agent import create_physics_agent
from logger import get_logger

load_dotenv()
VECTORSTORE = os.getenv('VECTORSTORE', 'vectorstore')

logger = get_logger(name='agent', log_file='agent.log')

def get_vector_store():
    ''' Returns vectorstore wether built or loaded '''
    raw_flag = os.getenv('build_vectorstore', 'false').lower()

    if raw_flag in ('true', '1', 'yes'):
        build_vectorstore = True
    elif raw_flag in ('false', '0', 'no'):
        build_vectorstore = False
    else:
        logger.error(f'Invalid build_vectorstore value: {raw_flag}. Defaulting to False.')
        build_vectorstore = False

    if not os.path.exists(VECTORSTORE):
        logger.info("Vector database doesn't exist yet. Initiate build.")
        build_vectorstore = True

    if not os.path.abspath(VECTORSTORE).startswith(os.getcwd()):
        raise RuntimeError("Unsafe vectorstore path")

    if build_vectorstore:
        logger.info('Scraping document and building vectorstore.')
        try:
            start = time.time()
            vectorstore = build_vectorstore_from_openstax(embeddings)
            build_duration = time.time() - start
            logger.info(f'Building vector database finished. Time: {build_duration:.2f}')
        except Exception as e:
            logger.error(f'Error scraping documents and building vectorstore: {e}')
            raise
    else:
        logger.info('Loading pre-built vector database.')
        try:
            start = time.time()
            vectorstore = FAISS.load_local(
                VECTORSTORE,
                embeddings,
                allow_dangerous_deserialization=True
            )
            load_duration = time.time() - start
            logger.info(f'Loading vector database finished. Time: {load_duration:.2f}')
        except Exception as e:
            logger.error(f'Error loading vectorstore: {e}')
            raise

    logger.info('Returning vectorstore.')
    return vectorstore

def agent_loop(llm, vectorstore):
    agent = create_physics_agent(llm, vectorstore)

    logger.info('Starting Physics Tutor AI Agent')
    print('Starting Physics Tutor AI Agent')
    print("-- input 'quit' to quit conversation\n\n")

    print(
        "Knowledge Attribution:\n"
        "This AI uses educational content from OpenStax – College Physics 2e.\n"
        "OpenStax is published by Rice University and licensed under CC BY 4.0.\n"
        "Content has been programmatically extracted, structured, and embedded for retrieval.\n"
    )

    # Agent introduction
    intro = agent.invoke(
        'Introduce yourself. What is your role, where can you help me, and what are your limitations.'
        )
    print(f'Physics AI Agent: {intro}')
    logger.info(f'Agent introduction: {intro}')

    while True:
        question = input('User: ')
        logger.info('User input received.')
        logger.info(f'User input: {question}')
        # to add input validation

        if question.strip().lower() in ('quit', 'exit', 'q'):
            print()
            time.sleep(0.5)
            print()
            time.sleep(0.5)
            print('Ending conversation.')
            logger.info('Ending conversation.')
            break

        result = agent.invoke(question)
        logger.info(f'Agent answer: {result}')
        print(f'Physics AI Agent: {result}')
    

if __name__ == '__main__':
    try:
        vectorstore = get_vector_store()
    except Exception as e:
        logger.error(f'Error occurred during vectorstore retrieval: {e}. Exiting agent conversation')
        vectorstore = None

    if vectorstore:
        agent_loop(llm, vectorstore)
    else:
        print('\n\nError occured during agent knowledge retrieval.')
        print()
        time.sleep(0.5)
        print()
        time.sleep(0.5)
        print('Exiting agent conversation.')

    