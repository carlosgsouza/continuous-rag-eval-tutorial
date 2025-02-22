import json
import os
import pytest

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from continuous_rag_eval.rag import RAG
import glob

@pytest.fixture(scope="module")
def eval_dataset():
  dataset = []
  for filepath in glob.glob("eval/data/*.json"):
    with open(filepath, "r") as f:
      data = json.load(f)
      dataset.extend(data)
  yield dataset

@pytest.fixture(scope="module")
def rag(eval_dataset):
  r = RAG()

  documents = []
  for q in eval_dataset:
    for c in q["context"]:
      doc = Document(
          page_content = c["page_content"],
          metadata = {"title": c["metadata"]["title"]}
      )
      documents.append(doc)
  r.store_documents(documents)

  return r

@pytest.fixture(scope="module")
def evaluation(eval_dataset, rag):
  """
  Evaluates the RAG model on the eval dataset and returns the aggregated scores.
  """

  # Executes a query for each entry in the eval dataset and stores the input and output data in
  # samples, which is passed to Ragas for evaluation..
  samples = []
  for q in eval_dataset:
    # Skips entries from the eval dataset that have just the context, but no questions.
    if "question" not in q:
      continue
    
    query = q["question"]
    result = rag.query(query)
    samples.append(
        SingleTurnSample(
            user_input=query,
            response=result["response"],
            retrieved_contexts=[
                doc.page_content for doc in result["source_documents"]
            ],
            reference=q["answer"],
        ))
  evaluation_dataset = EvaluationDataset(samples)

  # Initializes the metrics and the LLM used to calculate them.
  evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
  metrics = [
      # Generation metrics
      ResponseRelevancy(llm=evaluator_llm),
      Faithfulness(llm=evaluator_llm),
      # Retrieval metrics
      LLMContextPrecisionWithReference(llm=evaluator_llm),
      LLMContextRecall(llm=evaluator_llm)
  ]

  # Performs the actual evaluation and uploads the results to app.ragas.io.
  eval_result = evaluate(evaluation_dataset, metrics)
  try:
    dashboard_url = eval_result.upload()
    print(f"Ragas Dashboard: {dashboard_url}")
  except Exception as e:
    print(f"Failed to upload evaluation results: {e}")

  scores = eval_result.to_pandas()
  aggregated_scores = {
    metric_name: scores[metric_name].mean() for metric_name in [m.name for m in metrics]
  }
  os.makedirs("test-results", exist_ok=True)
  with open("test-results/aggregated_eval_metrics.json", "w") as f:
    json.dump(aggregated_scores, f, indent=2)

  return aggregated_scores

def test_answer_relevancy(evaluation):
  assert evaluation["answer_relevancy"] > 0.5

def test_faithfulness(evaluation):
  assert evaluation["faithfulness"] > 0.5

def test_llm_context_precision_with_reference(evaluation):
  assert evaluation["llm_context_precision_with_reference"] > 0.3

def test_context_recall(evaluation):
  assert evaluation["context_recall"] > 0.5

if __name__ == '__main__':
  pytest.main()
