# 🎼 Symphony Enterprise v2.0

**Multi-Agent Strategic Decision Intelligence Platform**

Symphony is an AI-powered pre-mortem analysis tool that simulates a rigorous, multi-perspective strategic debate. Four specialized AI agents provide comprehensive risk assessment for business plans and product ideas.

Built with **LangChain**, **LangGraph**, and **OpenAI**, featuring production-ready RAG (Retrieval-Augmented Generation) with vector search.

---

## ✨ Features

### 🤖 Four Specialized AI Agents
- **Shrek** - Internal Risk Architect (technical debt, over-engineering)
- **Sonic** - UX Friction Hunter (adoption velocity, user experience)
- **Hulk** - Market Adversary (competitive threats, market dynamics)
- **Trevor** - Strategic Synthesizer (final recommendations)

### 🔍 Advanced RAG Capabilities
- Upload PDF, TXT, DOCX documents to ground analysis in your company context
- Chroma vector store with OpenAI embeddings for semantic search
- Automatic document chunking and indexing
- Real-time context retrieval during agent analysis

### 🎯 Production-Ready Architecture
- **LangChain LCEL** chains for modular RAG pipelines
- **LangGraph StateGraph** for proper multi-agent orchestration
- **OpenAI API** integration with configurable models
- **Streaming output** for real-time user feedback

### 📊 Rich UI/UX
- Interactive executive dashboard with risk visualizations
- Real-time agent debate transcript
- Document upload and knowledge base management
- Risk matrix with Altair charts

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd LA_Musa
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.exemple .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.5
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Basic Analysis (Mock Data)
1. Keep "Use Mock Knowledge Base" checked in the sidebar
2. Enter your strategic plan or product idea
3. Click "Run Pre-Mortem Analysis"
4. Review the multi-agent debate and executive dashboard

### Advanced Analysis (Real RAG)
1. Uncheck "Use Mock Knowledge Base" in the sidebar
2. Upload company documents (PDFs, TXT, DOCX):
   - Company policies
   - Project post-mortems
   - Market research
   - Technical documentation
3. Click "Process Documents"
4. Enter your strategic plan
5. Run analysis - agents will cite your uploaded documents

### Understanding the Output

**Full Debate Transcript Tab:**
- Watch agents analyze in real-time
- Each agent provides evidence-based warnings
- Trevor synthesizes into actionable recommendations

**Executive Dashboard Tab:**
- High-level metrics and KPIs
- Risk matrix visualization
- Key insights and recommended actions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          Streamlit UI (app.py)              │
│  - File Upload  - Visualizations - Streaming│
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│    Orchestrator (LangGraph StateGraph)      │
│  - State Management  - Agent Routing        │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────────┐      ┌────────────────┐
│  Agents  │      │ Knowledge Base │
│ (LCEL)   │◄────►│   (Chroma)     │
└────┬─────┘      └────────────────┘
     │
     ▼
┌──────────────┐
│  LLM Client  │
│  (OpenAI)    │
└──────────────┘
```

### Key Components

**`src/orchestrator.py`**
- LangGraph StateGraph for multi-agent workflow
- Sequential execution: Shrek → Sonic → Hulk → Trevor
- State management with conversation history

**`src/agents.py`**
- LangChain LCEL chains
- RAG pipeline: retriever → prompt → LLM → parser
- Streaming and non-streaming execution

**`src/knowledge_base.py`**
- Chroma vector store integration
- Document loaders (PDF, TXT, DOCX)
- Semantic search with OpenAI embeddings

**`src/llm_client.py`**
- OpenAI API wrapper via LangChain
- Configurable models and parameters
- Error handling and retries

---

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Sampling temperature | `0.5` |
| `CHROMA_PERSIST_DIRECTORY` | Vector DB storage path | `./data/chroma_db` |

### Agent Configuration

Agents are configured in `config/agents.yaml`:
```yaml
agents:
  - name: "Shrek"
    prompt: "You are Shrek, the Guardian of the Swamp..."
  - name: "Sonic"
    prompt: "You are Sonic, the Friction Hunter..."
  # etc.
```

---

## 🧪 Testing

### Test with Mock Data (Fast)
```bash
# Use the checkbox in the UI - instant results with pre-populated data
```

### Test with Real Documents
1. Prepare sample documents (e.g., company policies, previous project docs)
2. Uncheck "Use Mock Knowledge Base"
3. Upload documents
4. Run analysis

### Sample Test Prompt
```
Launch a new multi-tenant SaaS offering on Azure for supply chain logistics,
using Cosmos DB for data storage, AKS for orchestration, and Entra ID B2C
for authentication. Target mid-market manufacturers with 100-1000 employees.
```

---

## 📦 Project Structure

```
LA_Musa/
├── app.py                      # Main Streamlit application
├── config/
│   └── agents.yaml            # Agent persona configurations
├── src/
│   ├── orchestrator.py        # LangGraph multi-agent orchestration
│   ├── agents.py              # Individual agent implementations
│   ├── knowledge_base.py      # RAG and vector store
│   └── llm_client.py          # OpenAI LLM client
├── data/
│   └── chroma_db/             # Vector database (auto-created)
├── requirements.txt           # Python dependencies
├── .env.exemple              # Environment template
├── Master.md                 # System design documentation
└── README.md                 # This file
```

---

## 🎯 Hackathon Criteria Alignment

### Industry Impact
- Solves universal problem: strategic risk assessment
- Saves 40+ hours of meetings and manual research
- Direct application in Product Management, Strategy, Marketing

### Responsible AI
- **Explainable**: Multi-agent debate exposes reasoning process
- **Grounded**: RAG ensures evidence-based analysis
- **Human-in-Loop**: Augments, doesn't replace decision-makers
- **Transparent**: All sources and reasoning visible

### Technical Excellence
- Production-ready LangChain/LangGraph architecture
- Modern RAG with vector search
- Real-time streaming output
- Modular, extensible design

---

## 🐛 Troubleshooting

### "OpenAI API Key not set"
- Ensure `.env` file exists with valid `OPENAI_API_KEY`
- Run `source .env` or restart the application

### Import errors for LangChain
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

### Chroma DB errors
- Delete `./data/chroma_db/` directory and restart
- Ensure write permissions in project directory

### Slow performance
- Use `gpt-4o-mini` instead of `gpt-4o` for faster responses
- Reduce document upload size
- Increase `max_tokens` in `.env` if responses are cut off

---

## 🚀 Deployment

### Streamlit Cloud (Browser Deployment)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on [streamlit.io](https://share.streamlit.io)**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path: `app.py`

3. **Configure Environment Variables in Streamlit Cloud**
   - In your Streamlit Cloud app settings, go to **Secrets**
   - Add the following secrets:
     ```toml
     OPENAI_API_KEY = "sk-your-actual-openai-api-key"
     OPENAI_MODEL = "gpt-4o-mini"
     OPENAI_TEMPERATURE = "0.50"
     CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
     ```

4. **Streamlit Configuration (streamlit_config.toml)**
   - The `streamlit_config.toml` is already configured for browser deployment
   - Key settings for browser environments:
     - `server.headless = true` - Runs without GUI
     - `server.address = "0.0.0.0"` - Binds to all interfaces
     - `browser.gatherUsageStats = false` - Disables telemetry

5. **Access your deployed app**
   - Your app will be available at: `https://your-app-name.streamlit.app`

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 💡 Future Enhancements

- [ ] Add more specialized agents (Legal, Finance, Operations)
- [ ] Implement agent voting/consensus mechanisms
- [ ] Add export to PDF/Word functionality
- [ ] Support for more document formats
- [ ] Integration with Azure AI Search
- [ ] Multi-language support
- [ ] Historical analysis comparison

---

## 🙏 Acknowledgments

Built for the **Microsoft AI Hackathon: Building the Future of Multi-Agent Systems**

Powered by:
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [Chroma](https://www.trychroma.com/)

---

**🎼 Symphony - Where AI Agents Debate, Strategy Emerges**
