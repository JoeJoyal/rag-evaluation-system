import os
from dotenv import load_dotenv
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_groq import ChatGroq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            max_tokens=800,
            temperature=0,
        )

    def load_model(self):
        return self.llm

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        model = self.load_model()

        if schema is not None:
            model = model.with_structured_output(schema)

        return model.invoke(prompt).content if schema is None else model.invoke(prompt)

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ):
        model = self.load_model()

        if schema is not None:
            model = model.with_structured_output(schema)

        result = await model.ainvoke(prompt)
        return result.content if schema is None else result


judge_model = GroqDeepEvalModel(
    model_name="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
)

test_case = LLMTestCase(
    input="Which company did Joe work for as a Generative AI Engineer, and during what dates?",
    actual_output=(
        "Joe worked as a Generative AI Engineer at Ramco cement factory"
        "in Dalmia Bharat, India, from August 2025 to September 2026."
    ),
    expected_output=(
        "Joe worked as a Generative AI Engineer at Rangsons Aerospace Pvt Ltd "
        "from August 2025 to September 2026."
    ),
)

metric = GEval(
    name="Correctness",
    criteria=(
        "Determine whether tha actual output is factually correct based on the expected output."
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=judge_model,
)

metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())