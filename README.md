# Antigravity Multi-Agent Research System 🚀

This repository contains a full-stack, AI-powered **Multi-Agent Research System**. By providing a target research topic, you can trigger a team of autonomous LangGraph agents to collaboratively browse the web, aggregate raw data, write a draft, fact-check it, and render a finalized markdown report in real-time.

## 🏗️ Architecture Stack

### Backend
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) for blazing-fast asynchronous endpoints.
- **AI Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://python.langchain.com/).
- **LLM Provider:** [Groq](https://groq.com/) using the `llama-3.3-70b-versatile` model for lightning-fast inference.
- **Search Tooling:** [Tavily](https://tavily.com/) for optimized web retrieval.
- **Streaming:** Server-Sent Events (SSE) via `sse_starlette` providing real-time UI logging of the LangGraph execution.

### Frontend
- **Interface:** Pure Vanilla JavaScript & HTML/CSS.
- **Styling:** Custom, highly responsive futuristic dark mode design featuring glassmorphism micro-interactions.
- **Rendering:** [Marked.js](https://marked.js.org/) for beautiful, out-of-the-box markdown conversion from the final AI report.

---

## 🤖 The Agent Workflow

The underlying LangGraph `StateGraph` defines three specialized nodes:

1. **Researcher Node:** Takes the user's initial query (or feedback loop instructions) and performs comprehensive web lookups via Tavily's Search API.
2. **Analyst Node:** Consumes the unstructured, raw web-scraped data to intelligently synthesize a structured Markdown draft.
3. **Reviewer Node:** Meticulously fact-checks the draft against the raw data.
   - *If satisfactory:* It yields an `[APPROVED]` signal and the final report.
   - *If data is missing/hallucinated:* It yields a `[REJECTED]` signal with specific feedback, seamlessly routing execution back to the **Researcher Node** for a new iteration.

---

## 💻 Installation & Setup

1. **Clone the repository.**
2. **Install system requirements:**
   Ensure you have Python 3.9+ available.
   ```bash
   pip install fastapi uvicorn langchain langchain-groq langgraph sse-starlette python-dotenv langchain-community tavily-python
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the project's root directory containing:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

---

## 🚀 How to Run Locally

You will need to run the **Backend API** and the **Frontend Web App** concurrently in two separate terminal windows.

### 1. Run the Backend (Terminal 1)
From the root directory, start the FastAPI server on port `8000`:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Run the Frontend (Terminal 2)
Navigate to the `frontend/` directory and spin up a lightweight Python HTTP server on port `3000`:
```bash
cd frontend
python -m http.server 3000
```

### 3. Test the App
Navigate to [http://localhost:3000](http://localhost:3000) in your web browser. Type a topic (e.g., *"Latest breakthroughs in Quantum Computing"*) and hit **Research**. Keep an eye on the live event log as the agents collaborate!
