from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain_community.embeddings import HuggingFaceEmbeddings

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.1'

@lru_cache
def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        dtype='auto'
    )

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
        max_new_tokens=512,
        temperature=0.2,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)

@lru_cache
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    return embeddings

