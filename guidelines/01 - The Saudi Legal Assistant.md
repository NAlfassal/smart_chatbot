# Project 1: The Saudi Legal Assistant

> Note: these are the minimum requirements for the project.

**Theme:** "Public Sector & Web Scraping"
**User:** General Public / Junior Legal Researchers.
**Goal:** An Arabic-first chatbot that answers questions based *strictly* on MOJ laws, providing citations.

* **Scope (MVP):**
* **Data:** **Web Scraper** required. Target specific laws (e.g., Labor Law, Corporate Law) from `laws.moj.gov.sa`.
* **Process:**
1. **Ingest:** Scrape  Clean  Chunk (by Article #)  Vector Store (ChromaDB).
2. **Retrieve:** Semantic search in Arabic.


* **Output:** Chat interface. Answers must cite the Article Number (e.g., "According to Article 77...").


* **Delivery Plan:** (draft)
* **Week 6:** **Scraping & Indexing.** Write scripts to crawl the site. Handle Arabic text normalization. Set up the Vector DB.
* **Week 7:** **RAG Pipeline.** optimize embeddings for Arabic (e.g., use multilingual models). Build the Chat UI.
* **Week 8:** **Guardrails.** Ensure the bot refuses to answer non-legal questions or give "legal advice" (add disclaimers).
