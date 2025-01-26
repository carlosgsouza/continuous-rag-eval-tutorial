import pytest

from langchain.schema import Document
from continuous_rag_eval.rag import RAG

@pytest.fixture
def rag():
  return RAG()

def test_query(rag):

  docs = [
      Document(
          page_content="Albert Einstein was a physicist who developed the theory of relativity. "
          "He was born in Germany in 1879 and died in 1955. "
          "Einstein is considered one of the most important scientists of all time. "
          "His work has had a major impact on our understanding of space, time, gravity, and the universe.",
          metadata={
              "title": "Albert Einstein"
          }
      ),
      Document(
          page_content="Isaac Newton was a physicist and mathematician who developed the laws of motion and universal gravitation. "
          "He was born in England in 1643 and died in 1727. "
          "Newton is considered one of the most important scientists of all time. "
          "His work laid the foundation for classical mechanics and had a profound impact on many fields of science.",
          metadata={
              "title": "Newton"
          }
      ),
  ]
  rag.stored_documents(docs)

  assert "Albert Einstein" in rag.query(query="Which physicist developed the theory of relativity?")["response"]
  assert "1643" in rag.query(query="When was Isaac Newton born?")["response"]

if __name__ == "__main__":
  pytest.main()
