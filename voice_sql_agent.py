import os
import speech_recognition as sr
import pyttsx3

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
            When the user asks for product details, query the database for answers.
            Present the answers in a nicely formatted tabular format.
            """,
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

model = "gpt-3.5-turbo"  # gpt-3.5-turbo, gpt-4o, "gpt-4", etc...


class VoiceAgent:
    def __init__(self):
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech engine
        print("Initializing text-to-speech engine...")
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed of speech

            # Test TTS
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)  # Use first available voice
            print("✓ Text-to-speech ready!")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            raise

        # Adjust for ambient noise
        print("Calibrating microphone for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✓ Calibration complete!")

    def listen(self):
        """Listen to microphone and convert speech to text"""
        with self.microphone as source:
            print("\nListening... (say 'quit' or 'exit' to stop)")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing speech...")

                # Using Google's speech recognition (free)
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.WaitTimeoutError:
                print("No speech detected. Please try again.")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

    def speak(self, text):
        """Convert text to speech"""
        print(f"\n🔊 AI: {text}\n")
        try:
            self.tts_engine.say(text)
            # Use startLoop(False) + iterate() pattern instead of runAndWait()
            # This properly waits for speech to complete
            self.tts_engine.startLoop(False)
            while self.tts_engine.isBusy():
                self.tts_engine.iterate()
            self.tts_engine.endLoop()
        except Exception as e:
            print(f"Error speaking: {e}")
            print("TTS failed, but text response is shown above.")


if __name__ == "__main__":
    # Obtain the API key from .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=0,
    )

    # Create SQL tools for the database
    db = SQLDatabase.from_uri("sqlite:///media_store.db")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)
    # This toolkit comes with 4 pre-defined tools
    toolkit = [*sql_toolkit.get_tools()[:4]]

    # Create the agent
    agent = create_openai_tools_agent(model, toolkit, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)  # Set to True for debugging
    message_history = ChatMessageHistory()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Initialize voice agent
    voice_agent = VoiceAgent()

    print("\n" + "="*50)
    print("Voice-Enabled Music Store Assistant")
    print("="*50)
    print("\nSpeak your questions to interact with the music store.")
    print("Say 'quit' or 'exit' to end the conversation.\n")

    # Run agent in an interactive voice loop
    while True:
        user_input = voice_agent.listen()

        if user_input is None:
            continue

        if user_input.lower() in ["quit", "exit", "stop"]:
            voice_agent.speak("Goodbye! Have a great day!")
            break

        # Get response from agent
        result = agent_with_chat_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "my_session_id"}},
        )

        # Speak the response
        voice_agent.speak(result["output"])
