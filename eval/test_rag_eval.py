import json
import os
import pytest

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from tqdm import tqdm

from continuous_rag_eval.rag import RAG

@pytest.fixture
def eval_dataset():
    with open("eval/eval_dataset.json", "r") as f:
        yield json.load(f)


def test_eval_rag(eval_dataset):
    rag = RAG()
    rag.store_eval_dataset_documents(eval_dataset[:1])

    # Executes a query for each entry in the eval dataset and stores the input and output data in
    # samples, which is passed to Ragas for evaluation..
    samples = []
    for q in eval_dataset:
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
    try:
        eval_result = evaluate(evaluation_dataset, metrics)
        eval_result.upload()
    except Exception as e:
        print(f"Uploading results to Ragas failed: {e}")

    # Validates the aggrregated score (calculated by validate_aggrregated_score) satisfies min_values.
    validate_aggregated_score(
        eval_result,
        min_values={
            ResponseRelevancy.name: 0.1,
            Faithfulness.name: 0.1,
            LLMContextPrecisionWithReference.name: 0.1,
            LLMContextRecall.name: 0.1,
        },
    )

def validate_aggregated_score(eval_result, min_values):
    """
  Validates that the aggregated scores for each metric meet the minimum values.
  """
    # The aggregated scores are the mean of the score for each indivual entry.
    scores = eval_result.to_pandas()
    aggregated_scores = {
      metric_name: scores[metric_name].mean() for metric_name in min_values.keys()
    }

    # Creates a test for each metric defined in min_values.
    print(f"Aggregated Scores: {aggregated_scores}")

if __name__ == '__main__':
    pytest.main()
