from langchain_core.runnables import RunnablePassThrough
from langchain_core.output_parsers import StrOutputParser

from prompt import physics_agent_prompt

def create_physics_agent(llm, vectorstore):
    ''' Creates physics agent chain'''
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k':3}
    )

    def format_docs(docs: list) -> str:
        return '\n\n'.join(doc.page_content for doc in docs)

    chain = (
        {
            'context': retriever | format_docs,
            'question': RunnablePassThrough()
        }
        | physics_agent_prompt
        | llm
        | StrOutputParser()
    )

    return chain