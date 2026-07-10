from  langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            # "You are a viral twitter influencer grading a tweet. Generate critique and recommendation for the user."
            # "Always provide the detailed recommendations, including requests for length, virality, style, etc."
            """
            You are an expert Twitter editor.

            Review ONLY the assistant's latest tweet.
            
            Do NOT rewrite it.
            
            Return only constructive criticism.

            Your response must be bullet points.
            
            Focus on:
            - Hook
            - Clarity
            - Virality
            - Length
            - Tone
            - Call to action
            """
        ),
        MessagesPlaceholder(variable_name='messages'),
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
            You are an expert Twitter/X writer.

            The first human message contains the original request.
            
            If later human messages contain feedback,
            treat them as critiques of your previous draft.
            
            Rewrite ONLY your previous tweet using the latest critique.
            
            Return only the tweet.
            """
        ),
        MessagesPlaceholder(variable_name='messages'),
    ]
)

llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0)
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm