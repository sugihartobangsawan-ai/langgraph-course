from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generations answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """
You are a grader assessing whether an LLM generation is grounded in / supported by 
a set of retrieved facts. 
Give a binary score 'yes' or 'no'. 
'Yes' means that the answer is grounded in / supported by the set of the facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human','set of facts: \n\n {documents} \n\n LLM generation: {generation}')

    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader