# 🎵 Agentic Music Store

An AI-powered music store assistant that demonstrates the power of agentic AI workflows using LangChain, OpenAI, and RAG (Retrieval-Augmented Generation). This project serves as a hands-on tutorial for building intelligent agents that can interact with databases, manage shopping carts, and retrieve contextual information.

## 🌟 Overview

This project showcases how to build an intelligent AI agent that combines multiple capabilities:

- **Database Querying**: Interact with a SQLite database (Chinook music store database) to search for songs, artists, and albums
- **Shopping Cart Management**: Add items to cart and view cart contents
- **RAG (Retrieval-Augmented Generation)**: Search through documents for store history and contextual information
- **Conversational Memory**: Maintain context across multiple interactions
- **Interactive UI**: Chat interface built with Streamlit

## 🏗️ Architecture

The project demonstrates a multi-tool agentic architecture:

```
User Input
    ↓
AI Agent (GPT-4o)
    ├── SQL Database Tools (4 tools)
    │   ├── Query database
    │   ├── Get schema
    │   ├── Execute queries
    │   └── Check query syntax
    ├── Shopping Cart Tools
    │   ├── add_to_cart
    │   └── show_cart
    └── RAG Tool
        └── document_search (Vector store retrieval)
```

## 🚀 Features

### 1. **SQL Database Integration**
- Query the Chinook music database containing tracks, albums, artists, and more
- Natural language to SQL conversion
- Schema inspection and query validation

### 2. **Shopping Cart Functionality**
- Add tracks to a shopping cart with prices
- View cart contents
- Persistent cart state during session

### 3. **RAG Document Retrieval**
- Load documents from multiple formats (PDF, TXT, DOCX, XLSX, CSV, PPTX)
- FAISS vector store for efficient similarity search
- OpenAI embeddings for document vectorization
- Retrieve store history and contextual information

### 4. **Conversational Memory**
- Maintain chat history across interactions
- Context-aware responses
- Session management

### 5. **Interactive UI**
- Streamlit-based chat interface
- Real-time responses
- Chat history display
- Clear chat functionality

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- LangSmith API key (optional, for tracing)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/algorythmist/agentic-musicstore.git
cd agentic-musicstore
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
```

5. **Prepare the database**

Ensure you have the `chinook.db` SQLite database in the project root. This is the standard Chinook sample database containing music store data.

6. **Add documents (optional)**

Place any documents you want the agent to reference in a `documents/` directory.

## 🎮 Usage

### Running the Streamlit UI

```bash
streamlit run user_interface.py
```

This will launch a web interface where you can interact with the AI assistant.

### Running the CLI Agent

For a command-line interface experience:

```bash
python chinook_musicstore_agent.py
```

### Running the Simple SQL Agent

To test just the SQL functionality:

```bash
python simple_sql_agent.py
```

## 💡 Example Interactions

**Querying the Database:**
```
User: "What albums does AC/DC have?"
AI: [Queries database and returns AC/DC albums]
```

**Adding to Cart:**
```
User: "Add the song 'Thunderstruck' to my cart for $1.99"
AI: "Thunderstruck has been added to the cart."
```

**Document Retrieval:**
```
User: "Tell me about the store's history"
AI: [Retrieves relevant information from documents]
```

**Showing Cart:**
```
User: "Show me my cart"
AI: [Displays all items in the cart]
```

## 📁 Project Structure

```
agentic-musicstore/
├── cart_tool.py                    # Shopping cart tool implementation
├── chinook_musicstore_agent.py     # Main agent with all tools
├── documents.py                     # Document loading utilities
├── rag_tool.py                      # RAG tool with vector store
├── simple_sql_agent.py              # Simplified SQL-only agent
├── user_interface.py                # Streamlit UI
├── requirements.txt                 # Project dependencies
├── chinook.db                       # SQLite music database
└── documents/                       # Directory for RAG documents
```

## 🔧 Key Components

### 1. Cart Tool (`cart_tool.py`)
Implements a simple shopping cart using LangChain's `@tool` decorator:
- `add_to_cart(item_name, item_price)`: Add items to cart
- `show_cart()`: Display cart contents

### 2. RAG Tool (`rag_tool.py`)
Builds a retrieval tool using:
- **FAISS**: Facebook's vector similarity search library
- **OpenAI Embeddings**: Text embedding model (`text-embedding-3-large`)
- **Document Loader**: Supports multiple file formats

### 3. Document Loader (`documents.py`)
Utility for loading and chunking documents:
- Supports: PDF, TXT, DOCX, XLSX, CSV, PPTX
- Uses `CharacterTextSplitter` for chunking
- Handles both single files and directories

### 4. Main Agent (`chinook_musicstore_agent.py`)
Orchestrates all tools:
- **SQL Toolkit**: 4 database tools from LangChain
- **Cart Tools**: Shopping cart operations
- **RAG Tool**: Document retrieval
- **Message History**: Conversation memory
- **LangSmith Tracing**: Optional observability

## 🎓 Learning Objectives

This tutorial project teaches:

1. **Agent Architecture**: How to build multi-tool agents with LangChain
2. **Tool Integration**: Combining different tool types (SQL, custom tools, RAG)
3. **Prompt Engineering**: Crafting effective system prompts for agents
4. **Memory Management**: Implementing conversational context
5. **RAG Implementation**: Building retrieval systems with vector stores
6. **UI Development**: Creating chat interfaces with Streamlit
7. **Observability**: Using LangSmith for debugging and tracing

## 🔍 How It Works

### Agent Flow

1. **User Input**: User asks a question via CLI or UI
2. **Agent Processing**: The agent analyzes the input and determines which tool(s) to use
3. **Tool Execution**: 
   - For product queries → SQL tools
   - For cart operations → Cart tools
   - For store info → RAG tool
4. **Response Generation**: Agent synthesizes results into a natural language response
5. **Memory Update**: Conversation history is updated for context

### Tool Selection Logic

The agent uses OpenAI's function calling to intelligently select tools based on:
- Tool descriptions
- User intent
- Conversation context
- System prompt guidance

## 📊 Database Schema

The Chinook database includes these main tables:
- `Artist`: Artist information
- `Album`: Albums by artists
- `Track`: Individual tracks with pricing
- `Customer`: Customer information
- `Invoice`: Purchase records
- `Genre`: Music genres
- `MediaType`: Track formats

## 🔐 Security Considerations

- Store API keys in `.env` file (never commit to version control)
- Use environment variables for sensitive data
- The `.env` file should be added to `.gitignore`

## 🐛 Troubleshooting

### Common Issues

**"No module named 'faiss'"**
```bash
pip install faiss-cpu
```

**"Database not found"**
- Ensure `chinook.db` is in the project root
- Download from: [Chinook Database](https://github.com/lerocha/chinook-database)

**"OpenAI API Error"**
- Verify your API key in `.env`
- Check your OpenAI account has sufficient credits

**"Documents not loading"**
- Ensure documents are in the correct format
- Check file permissions
- Verify the `documents/` directory exists

## 🚀 Next Steps & Extensions

Ideas for extending this project:

1. **Enhanced Cart Features**
   - Remove items from cart
   - Calculate totals with tax
   - Checkout functionality

2. **Additional Tools**
   - Recommendation engine
   - User preferences storage
   - Order history tracking

3. **Improved RAG**
   - Multiple vector stores
   - Hybrid search (keyword + semantic)
   - Document metadata filtering

4. **Better UI**
   - Display cart in sidebar
   - Show database query results in tables
   - Add audio previews

5. **Production Readiness**
   - User authentication
   - Database connection pooling
   - Error handling and logging
   - Rate limiting

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chinook Database](https://github.com/lerocha/chinook-database)

## 🤝 Contributing

This is a tutorial project. Feel free to:
- Fork and experiment
- Submit issues for clarification
- Share your extensions and improvements

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- Chinook Database for the sample music store data
- LangChain for the agent framework
- OpenAI for the language models
- Streamlit for the UI framework

---

**Happy Building! 🎵🤖**

For questions or feedback, please open an issue on GitHub.