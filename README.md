# 🤖 Agentic AI - Multi-Agent Research System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io)

A production-ready multi-agent research system powered by **LangChain**, **LangGraph**, and **NVIDIA NIM**. Features real-time web search, Wikipedia integration, ArXiv academic papers, PDF analysis, persistent memory, and export to PDF/Word.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Architecture** | Planner → Researcher → Writer → Critic workflow |
| **Multi-Source Research** | DuckDuckGo + Wikipedia + ArXiv academic papers |
| **NVIDIA NIM Integration** | Llama 3.3 70B via OpenAI-compatible API |
| **PDF Analysis** | Upload and analyze documents with AI |
| **Persistent Memory** | SQLite database for research history |
| **Export Options** | Download reports as PDF or Word (DOCX) |
| **Streamlit Dashboard** | Interactive web UI with sidebar navigation |
| **Batch Research** | Research multiple topics simultaneously |

---

## 📋 Requirements

```bash
pip install -r requirements.txt
```

---

## Setup 

Set your NVIDIA API key as an environment variable:
```
# Windows CMD
set NVIDIA_API_KEY=nvapi-your-key-here

# Windows PowerShell
$env:NVIDIA_API_KEY="nvapi-your-key-here"

# Or create a .env file
echo NVIDIA_API_KEY=nvapi-your-key-here > .env
```
Get your free NVIDIA API key at [build.nvidia.com](build.nvidia.com) 

## Usage
**CLI Mode**
```
python multi_agent_research_system_enhanced.py
```
**Streamlit Dashboard**
```
streamlit run multi_agent_research_system_enhanced.py --server.fileWatcherType none
```
Then open http://localhost:8501 in your browser.

## 🏗️ Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────▶│  Researcher │────▶│   Writer    │
│   Agent     │     │   Agent     │     │   Agent     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Critic    │
                                        │   Agent     │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │  Final      │
                                        │  Report     │
                                        └─────────────┘
```

## 📁 Project Structure

```
agentic-ai-research-system/
├── multi_agent_research_system_enhanced.py   # Main application
├── requirements.txt                          # Dependencies
├── .gitignore                               # Git ignore rules
├── README.md                                # This file
├── LICENSE                                  # MIT License
└── research_memory.db                       # Auto-created SQLite DB
```

## 🛠️ Tech Stack

- **LangChain** - LLM framework
- **LangGraph** - Agent orchestration
- **NVIDIA NIM** - LLM inference (Llama 3.3)
- **Streamlit** - Web UI
- **SQLite** - Persistent storage
- **DuckDuckGo** - Web search
- **Wikipedia** - Knowledge base
- **ArXiv** - Academic papers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Varnit Rana** - [GitHub](https://github.com/ranavarnit)
