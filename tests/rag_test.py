import pytest

from langchain.schema import AIMessage, Document
from continuous_rag_eval.rag import RAG


@pytest.fixture
def test_docs():
  return [
      Document(
          page_content="Albert Einstein was a physicist who developed the theory of relativity.",
          metadata={"title": "Albert Einstein"}
      ),
      Document(
          page_content="He was born in Germany in 1879 and died in 1955.",
          metadata={"title": "Albert Einstein"}
      )
  ]


def test_store_documents(mocker, test_docs):
  vector_store = mocker.MagicMock()
  rag = RAG(vector_store=vector_store)

  rag.stored_documents(test_docs)

  vector_store.add_documents.assert_called_once_with(test_docs)


def test_query(mocker, test_docs):
  llm = mocker.MagicMock(name="llm")
  vector_store = mocker.MagicMock(name="vector_store")
  
  rag = RAG(llm=llm, vector_store=vector_store)

  query = "Who was Albert Einstein?"
  response = "He was a physicist who developed the theory of relativity"
  
  vector_store.as_retriever.return_value.invoke.return_value = test_docs
  llm.invoke.return_value = AIMessage(content=response)

  result = rag.query(query)

  assert result["response"] == response
  assert result["source_documents"] == test_docs


if __name__ == "__main__":
  pytest.main()