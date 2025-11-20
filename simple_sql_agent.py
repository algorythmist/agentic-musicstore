import os

from dotenv import load_dotenv
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI assistant for a Music Store.
            You help customers explore songs, artists, and albums through friendly, interactive questions.
            When the user asks for product details, you can query the database.
            """,
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

if __name__ == "__main__":
    # Obtain the API key from .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model="gpt-4o", api_key=openai_api_key  # or "gpt-4", "gpt-3.5-turbo", etc.
    )

    # Create SQL tools for the database
    db = SQLDatabase.from_uri("sqlite:///media_store.db")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)
    # This toolkit comes with 4 pre-defined tools
    toolkit = [*sql_toolkit.get_tools()[:4]]

    # Create the agent
    agent = create_openai_tools_agent(model, toolkit, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
    message_history = ChatMessageHistory()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Run agent in an interactive loop
    while True:
        user_input = input("Type your question here (or type 'q' to quit): ")
        if user_input.lower() == "q":
            break
        result = agent_with_chat_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "my_session_id"}},
        )
        print(f"User: {user_input}")
        print(f'AI: {result["output"]}')
