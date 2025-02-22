import glob
import json
import os

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from continuous_rag_eval.rag import RAG

def run_evaluation():
  eval_dataset = load_eval_dataset()
  rag = initialize_rag(eval_dataset)

  # Executes a query for each entry in the eval dataset and stores the input and output data in
  # samples, which is passed to Ragas for evaluation..
  samples = []
  # TODO: Evaluate the entire dataset once it is ready.
  for q in eval_dataset[:3]:
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
  results_url = eval_result.upload()
  print(f"Results dashboard: {results_url}")

  # Exports the aggregated metrics to a JSON file.
  scores = eval_result.to_pandas()
  aggregated_scores = {
    metric_name: scores[metric_name].mean() for metric_name in [m.name for m in metrics]
  }
  
  os.makedirs("eval-results", exist_ok=True)
  with open("eval-results/aggregated_metrics.json", "w") as f:
    json.dump(aggregated_scores, f, indent=2)

def load_eval_dataset():
  dataset = []
  for filepath in glob.glob("eval/data/*.json"):
    with open(filepath, "r") as f:
      data = json.load(f)
      dataset.extend(data)
  return dataset

def initialize_rag(eval_dataset):
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

 
if __name__ == '__main__':
  run_evaluation()
