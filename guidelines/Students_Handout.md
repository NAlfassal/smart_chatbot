# 🎓 Capstone Project Guide: Tracks & Rubric

* **Timeline:** 3 Weeks (Weeks 6-8)
* **Team Size:** 2-4 Members
* **Goal:** Build a production-grade AI application.

---

## 🚀 Choose Your Track

Select **ONE** of the following three real-world scenarios for your project.

### **Track A: The AI Recruiter (HR Tech)**

* **Scenario:** HR teams are drowning in PDFs. They need a system to screen candidates instantly.
* **The Build:** An app that takes **Resume PDFs** and a **Job Description**, extracts key data into structured JSON, and outputs a "Fit Score" with reasoning.
* **Key Challenge:** **Reasoning.** The AI must justify *why* a candidate is a good/bad fit (e.g., "Missing Python skill").
* **Primary Tech:** Prompt Engineering, JSON Output Parsers, file processing.

### **Track B: The Legal Assistant (Arabic RAG)**

* **Scenario:** Legal researchers need accurate answers from Ministry of Justice laws without manual searching.
* **The Build:** A chatbot that answers questions based on data scraped from `laws.moj.gov.sa`.
* **Key Challenge:** **Accuracy.** Every answer must cite the specific Article Number. No hallucinations allowed.
* **Primary Tech:** Web Scraping, Vector Databases, Arabic NLP.

### **Track C: The Data Analyst (Text-to-SQL)**

* **Scenario:** Executives want to query their database using natural language, not SQL code.
* **The Build:** An agent that translates questions ("Show me top sales in Riyadh") into SQL queries, executes them, and visualizes the results.
* **Key Challenge:** **Reliability.** The agent must self-correct if the generated SQL is invalid.
* **Primary Tech:** LangChain/LlamaIndex Agents, SQL Toolkits, Data Visualization.

---

## ⚖️ Grading Rubric (Total: 100 Points) - **Tentative**

### **Part 1: Common Core (60 Points)**

*Applies to ALL teams.*

| Criteria | Points | Requirement |
| --- | --- | --- |
| **Architecture** | **20** | Clean code organization (modular functions, not one giant script). Clear separation of Data vs. App logic. |
| **UX / UI** | **20** | User-friendly interface (Streamlit/Gradio). Handles errors gracefully. Shows "Thinking..." status. |
| **Presentation** | **20** | Clear problem/solution story. **The Live Demo must work.** |

### **Part 2: Track-Specific Success (40 Points)**

*Your team is graded on the specific goals of your chosen track.*

| Track | Criteria (20 pts) | Criteria (20 pts) |
| --- | --- | --- |
| **A: Recruiter** | **Structured Extraction:** Accurately parses messy PDFs into clean JSON (Skills, YOE). | **Logic & Reasoning:** Explains the "Why" behind the match score clearly. |
| **B: Legal Bot** | **Citation Quality:** Answers always cite sources (e.g., "Source: Article 77"). | **Arabic Handling:** Retrieval and UI work seamlessly in Arabic. |
| **C: Analyst** | **Self-Correction:** System catches and fixes its own SQL syntax errors automatically. | **Visualization:** Generates charts (Bar/Line) for relevant queries, not just text/tables. |

> *Note: The rubric is tentative and may be adjusted.*

---

## Choose Your Team

### Range

**Range**: 2-4 members per team.

### For 3+ Members

If your team has **3 or 4 members**, you must include **at least TWO** of the following advanced features to pass:

1. **Hybrid Search:** Combine Keyword search + Vector search for better accuracy.
2. **Evaluation Pipeline:** Automated testing using Ragas/DeepEval to score your app.
3. **Deployment:** The app is deployed to the cloud (Render/Streamlit Cloud), not just localhost.
4. **Multi-Modality:** Processing images or complex tables within documents.

Also:

* **Code Quality:** Stricter requirements for modularity. Code must be organized in `src/` folders, not a single `app.py` script.
* **Documentation:** A comprehensive `README.md` is insufficient. They must provide a **System Architecture Diagram** and an **Evaluation Report** showing accuracy metrics.

### Role Specialization

For larger teams, generic "full-stack" roles cause friction. Assign specific ownership:

* **Member 1: Data & Embedding Engineer.** Handles ETL, chunking strategies, and vector DB management.
* **Member 2: AI Engineer.** Focuses on prompt engineering, agent tools, and LLM orchestration (LangChain/LlamaIndex).
* **Member 3: Frontend & Product.** Builds the UI and manages user experience/state.
* **Member 4 (if applicable): MLOps & Eval.** Implementation of evaluation frameworks (Ragas), logging, and deployment (Docker/Cloud).

## Project Proposal Submission

**Due**: Thursday, 15th January.

See the [**Capstone Proposal Evaluation Rubric**](Proposal_Evaluation.md) for details.

## Next Weeks

- **Phase 1**: The MVP Sprint (Week 6 - Building AI Apps)
- **Phase 2**: Agentic Intelligence (Week 7 - Agentic AI)
- **Phase 3**: The Final Polish (Week 8 - Capstone Sprint)
