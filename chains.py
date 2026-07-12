from  langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            You are a Twitter editor.

            Review this tweet.
            
            Return only bullet point feedback.
            """
        ),
        (
            'human',
            '{draft}'
        )
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            # "You are a twitter  techie influencer assistant tasked with writing excellent twitter posts."
            # "Generate the best twitter post possible for the user's request"
            # "If the user provides the critique, respond with a revised version of your previous attempts."
            """
            You are an expert Twitter writer.

            Improve the draft using the critique.
            
            If the draft is empty, write a brand new tweet.
            
            Return only the tweet.
            """
        ),
        ("human",
         """
        Original request:
        {original_request}
        
        Previous draft:
        {draft}
        
        Critique:
        {critique}
         """
         )
    ]
)

llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0)
# llm = ChatOllama(model='qwen3.5:9b')

generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm