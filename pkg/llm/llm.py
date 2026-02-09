# pip install --upgrade --quiet  langchain langchain-core langchain-community langchain-text-splitters langchain-milvus langchain-openai bs4

import os

os.environ["OPENAI_API_KEY"] = "sk-***********"

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus, Zilliz

# Create a WebBaseLoader instance to load documents from web sources
urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/", "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"]
# urls = ["https://confluence.shopee.io/pages/viewpage.action?pageId=2337522062"]
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
# Load documents from web sources using the loader
documents = loader.load()


# Initialize a RecursiveCharacterTextSplitter for splitting text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # todo 这两个参数是关键

# Split the documents into chunks using the text_splitter
docs = text_splitter.split_documents(documents)

# Let's take a look at the first document
# for doc in docs:
#     print(doc)


from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings() # todo 可以使用其他的embedding模型
bge_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"},  # 或 "cuda" 如果有GPU
    encode_kwargs={
        "normalize_embeddings": True  # BGE模型需归一化向量以使用余弦相似度
    }
)

vectorstore = Milvus.from_documents(  # or Zilliz.from_documents
    documents=docs,
    embedding=bge_embeddings,
    connection_args={
        "uri": "http://localhost:19530",
        "collection": "magao_test",
        "collection_properties": {
            "metric_type": "COSINE"  # 必须配置为余弦相似度
        }
    },
    drop_old=True,  # Drop the old Milvus collection if it exists
)

query = "what is: Subgoal and decomposition?"
# results = vectorstore.similarity_search(query, k=1)
# print(results)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # todo 不一定是openAI，也可以是deepseek

# Initialize the OpenAI language model for response generation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()


# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# rag_chain.get_graph().print_ascii()

# Invoke the RAG chain with a specific question and retrieve the response
res = rag_chain.invoke(query)









# res
#
# vectorstore.similarity_search(
#     "What is CoT?",
#     k=1,
#     expr="source == 'https://lilianweng.github.io/posts/2023-06-23-agent/'",
# )
#
# from langchain_core.runnables import ConfigurableField
#
# # Define a new retriever with a configurable field for search_kwargs
# retriever2 = vectorstore.as_retriever().configurable_fields(
#     search_kwargs=ConfigurableField(
#         id="retriever_search_kwargs",
#     )
# )
#
# # Invoke the retriever with a specific search_kwargs which filter the documents by source
# retriever2.with_config(
#     configurable={
#         "retriever_search_kwargs": dict(
#             expr="source == 'https://lilianweng.github.io/posts/2023-06-23-agent/'",
#             k=1,
#         )
#     }
# ).invoke(query)
#
# rag_chain2 = (
#         {"context": retriever2 | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
# )
#
# rag_chain2.with_config(
#     configurable={
#         "retriever_search_kwargs": dict(
#             expr="source == 'https://lilianweng.github.io/posts/2023-06-23-agent/'",
#         )
#     }
# ).invoke(query)