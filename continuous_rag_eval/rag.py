from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langsmith import traceable

class RAG:
  """
  Retrieval-Augmented Generation (RAG) class for question answering.
  """

  def __init__(self, llm=None, vector_store=None):
    """
    Initializes the LLM and and the vector store.
    """

    self.llm = llm if llm is not None else ChatOpenAI(temperature=0)
    self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore(
        embedding=OpenAIEmbeddings())
    
  def store_documents(self, documents):
    """
    Stores documents in the vector store.
    """
    self.vector_store.add_documents(documents)

  @traceable(name="RAG Query")
  def query(self, query: str) -> dict:
    """
    Queries the vector DB and uses the LLM to answer the query. 
    
    Returns a dict with the response and the source documents.
    """
    
    docs = self.vector_store.as_retriever().invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt_template = """You are an assistant for question-answering tasks. Use the following pieces
    of retrieved context to answer the question. If the answer is not in the context, just say "I 
    don't know". Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    Context: {context} 
    Answer:"""

    prompt = prompt_template.format(context=context, question=query)
    response = self.llm.invoke(prompt).content

    return {
      "response": response,
      "source_documents": docs
    }