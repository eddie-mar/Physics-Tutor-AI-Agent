from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain_community.embeddings import HuggingFaceEmbeddings

model_id = 'mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype='auto'
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

llm = HuggingFacePipeline(pipeline=pipe)

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)


