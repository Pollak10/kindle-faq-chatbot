# KindleBot — Kindle FAQ Chatbot

KindleBot is a conversational FAQ assistant for Amazon Kindle. You ask it a question about your Kindle — setup, charging, buying books, troubleshooting — and it finds the most relevant support information and gives you a plain-English answer.

It uses a RAG (Retrieval-Augmented Generation) pipeline: instead of just prompting an LLM and hoping it remembers the right answer, it first searches a knowledge base of Kindle FAQ content, pulls the most relevant pieces, and hands them to Claude as context. The result is answers that are grounded in actual documentation rather than whatever the model feels like saying.

---

## What it covers

The knowledge base has 57 Q&A pairs across six support categories:

- Setup & Registration
- Battery & Charging
- Content & Books
- Account & Subscriptions
- Reading Features
- Troubleshooting

---

## How it works

When you type a question, three things happen:

1. Your question gets converted into a vector embedding using a local sentence-transformers model
2. ChromaDB searches the knowledge base for the three most semantically similar FAQ entries
3. Those entries get passed to Claude Haiku as context, and Claude writes a response based only on what was retrieved

The key word is *only* — the system prompt explicitly tells Claude not to make things up if the retrieved context does not cover the question. It redirects to Amazon support instead. This was an intentional design choice to avoid hallucination on a support-focused chatbot.

---

## Stack

| Component | Tool |
|---|---|
| LLM | Anthropic Claude Haiku (`claude-haiku-4-5-20251001`) |
| Orchestration | LangChain (LCEL chain) |
| Vector store | ChromaDB (persisted locally) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Language | Python 3.9+ |

I went with a local embedding model rather than an API-based one (like OpenAI Embeddings) to avoid a second paid dependency. The HuggingFace model downloads once (~90 MB) and runs on CPU.

---

## Running it

**Requirements:** Python 3.9+, an Anthropic API key ([get one here](https://console.anthropic.com))

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/kindle-faq-chatbot.git
cd kindle-faq-chatbot
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"   # Mac/Linux
# set ANTHROPIC_API_KEY=your-key-here      # Windows CMD

# Run
python kindle_faq_chatbot.py
```

The first run takes about 30-60 seconds to download the embedding model and build the ChromaDB index. After that it starts instantly because the index is saved locally to `./kindle_chroma_db/`.

**Example:**
```
You: my kindle won't charge
KindleBot: Try a different USB cable first — cables are the most common failure
point. Clean the USB port gently and use a wall adapter rather than a computer
USB port. If the screen shows a critical battery icon, leave it plugged in for
30 minutes before pressing the power button.

  [Sources consulted: Battery & Charging]
```

Type `quit` or `exit` to close.

---

## What worked, what didn't

Semantic search handled paraphrased questions better than I expected. Asking "Kindle won't turn on" correctly retrieved the right FAQ entries even though none of them use those exact words. The cosine similarity approach generalises well to the natural variation in how people phrase support questions.

The main limitation is the size of the knowledge base. Fifty-seven entries covers the most common questions but there are gaps — anything outside those topics gets redirected to Amazon support. Scraping the full Kindle help centre would fix this but felt out of scope for the assignment.

A few things I would add with more time:

- **Conversation memory** so follow-up questions work properly ("How do I do that on Windows?")
- **A web interface** using Gradio or Streamlit so it runs in a browser without needing a terminal
- **Evaluation metrics** using RAGAS to measure retrieval precision and answer faithfulness systematically rather than just testing it manually

---

## Dataset

Curated from Amazon's official Kindle help pages:
https://www.amazon.com/gp/help/customer/display.html?nodeId=GDGXNMWD9KDPZVKE

57 records, 3 features per record (`question`, `answer`, `category`).

---

## References

Amazon. (2024). *Kindle help and customer service*. https://www.amazon.com/gp/help/customer/display.html?nodeId=GDGXNMWD9KDPZVKE

Anthropic. (2024). *Claude API documentation*. https://docs.anthropic.com

Chase, H. (2022). *LangChain* [Software library]. https://www.langchain.com

Chroma. (2023). *ChromaDB: The AI-native open-source embedding database*. https://www.trychroma.com

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459-9474. https://arxiv.org/abs/2005.11401

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. https://arxiv.org/abs/1908.10084
