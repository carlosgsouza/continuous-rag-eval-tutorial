from langchain import hub
from langchain.callbacks import LangChainTracer
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langsmith import traceable

class RAG:
  """
  Retrieval-Augmented Generation (RAG) class for question answering.
  """

  def __init__(self):
    """
    Initializes RAG with the default LLM, embeddings, and an empty in-memory vector store.
    """

    self.llm = ChatOpenAI(temperature=0)
    self.embedding = OpenAIEmbeddings()
    self.vector_store = InMemoryVectorStore(embedding=self.embedding)
    
    rag_prompt = hub.pull("rlm/rag-prompt")
    self.rag_chain = (
        rag_prompt
      | self.llm
      | StrOutputParser()
    )
    
  def stored_documents(self, documents):
    """
    Stores documents in the vector store.
    """
    self.vector_store.add_documents(documents)
    
  def store_eval_dataset_documents(self, eval_dataset: list):
    """
    Extracts the context entries from the eval dataset and stores them in the vector store. This is
    a convenience method to simplity the evaluation code.
    """
    documents = []
    for q in eval_dataset:
      for c in q["context"]:
        doc = Document(
            page_content = c["page_content"],
            metadata = {"title": c["metadata"]["title"]}
        )
        documents.append(doc)
    self.stored_documents(documents)

  @traceable(name="RAG Query")
  def query(self, query: str) -> dict:
    """
    Queries the vector DB and uses the LLM to answer the query. 
    
    Returns a dict with the response and the source documents.
    """
    
    # We manually handle tracing so separate calls to the retriever and the LLM are logged
    # together in LangSmith.
    
    docs = self.vector_store.as_retriever().invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    response = self.rag_chain.invoke({"context": context, "question": query})


    return {
      "response": response,
      "source_documents": docs
    }