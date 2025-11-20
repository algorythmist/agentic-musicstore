import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tracers import LangChainTracer
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith import uuid7

from cart_tool import add_to_cart, show_cart
from rag_tool import build_rag_tool


def get_model(openai_api_key: str, model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, api_key=openai_api_key)


def get_database_toolkit(database_uri: str, model) -> SQLDatabaseToolkit:
    db = SQLDatabase.from_uri(database_uri)
    return SQLDatabaseToolkit(db=db, llm=model)


def build_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_runnable_agent(model, prompt, toolkit):
    agent = create_openai_tools_agent(model, toolkit, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
    message_history = ChatMessageHistory()
    return RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def build_agent(openai_api_key):
    model = get_model(openai_api_key, "gpt-4o")  # or "gpt-4", "gpt-3.5-turbo", etc.)
    # TODO describe the 4 tools that come into this
    sql_toolkit = get_database_toolkit("sqlite:///chinook.db", model)
    rag_tool = build_rag_tool(
        openai_api_key,
        "documents",
        "document_search",
        """
            Search and return information about music store's and owner's history.
            """,
    )
    toolkit = [*sql_toolkit.get_tools()[:4], add_to_cart, show_cart, rag_tool]
    # TODO: Improve
    prompt = build_prompt(
        """
        You are an AI assistant for a Music Store.
        You help customers explore songs, artists, and albums through friendly, interactive questions.
        When the user asks for product details, you can query the database.
        When the user wants to buy a track, you can use the cart tool.
        When the user asks about details from the store, use the document retrieve tool 
        """
    )
    return build_runnable_agent(model, prompt, toolkit)


# TODO: Convert to UI
if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    agent_with_chat_history = build_agent(openai_api_key)
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    tracer = LangChainTracer(client=client, project_name="chinook")

    # Interactive loop
    session_id = uuid7()
    while True:
        user_input = input("Type your question here (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result = agent_with_chat_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}, "callbacks": [tracer]},
        )
        print(f"User: {user_input}")
        print(f'AI: {result["output"]}')
