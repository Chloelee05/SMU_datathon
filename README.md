# 🚢 SMU Cargill Datathon 2026 – Multi-Agent Chatbot UI

A modern **ChatGPT-style chatbot interface** built with **Streamlit**, powered by **LangChain + LangGraph multi-agent architecture**, developed for **SMU Cargill Datathon 2026**.

The system integrates multiple specialised AI agents (optimisation, math, supervision) with tool-calling to support **querying, reasoning, and calculations** in a clean web UI.

---

## 🧠 Key Features

- 💬 ChatGPT-style UI built with Streamlit  
- 🤖 Multi-agent system using LangChain + LangGraph  
- 🧮 Tool-calling agents for:
  - Mathematical reasoning  
  - Optimisation logic  
  - Supervised task routing  
- 🔁 Persistent chat state across interactions  
- 🔐 Secure API key handling via environment variables or Streamlit secrets  
- ⚡ Cached agent graph for fast responses  

---

## 📸 Screenshots

*(Add screenshots here once final UI is ready)*

```text
screenshots/
├── chat_ui.png
├── agent_response.png
└── tool_call_trace.png
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** – frontend & app server
- **LangChain** – LLM orchestration
- **LangGraph** – multi-agent control flow
- **OpenAI API** – LLM backend

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Chloelee05/SMU_datathon.git
cd SMU_datathon
```

### 2️⃣ Create and activate a virtual environment (recommended)

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Setting the OpenAI API Key
###  Environment Variable (PowerShell)

```powershell
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

Verify:
```powershell
echo $env:OPENAI_API_KEY
```

---

## ▶️ Running the Application

From the project root:

```bash
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

---

## 🗂️ Project Structure

```text
├── app.py                   # Main Streamlit application
├── supervisor_agent.py      # Supervisor / router agent
├── optimiser_agent.py       # Optimisation agent
├── math_agent.py            # Math & calculation agent
├── z_data_in.py             # Data loading / preprocessing
├── z_calculator_latest.py   # Domain-specific calculators
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ How It Works (High-Level)

1. User enters a query in the Streamlit chat UI  
2. Input is passed to a **LangGraph supervisor agent**  
3. Supervisor routes the task to:
   - Math agent  
   - Optimiser agent  
   - Or other tools  
4. Agents call tools when needed  
5. Final response is returned to the UI  

---

## 🧪 Development Notes

- Agents are initialised lazily to avoid API key issues  
- LangGraph is cached using `st.cache_resource`  
- Streamlit reruns are handled safely with session state  

---

## 📝 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Cargill & SMU** – Datathon 2026  
- **OpenAI** – Large Language Models  
- **LangChain / LangGraph** – Agent orchestration  
- **Streamlit** – UI framework  
