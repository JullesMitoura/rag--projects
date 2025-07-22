## RAG, GraphRAG and AgenticRAG Projects

### Project 01:
#### RAG with AzureOpenAI and FAISS
---
The first project in this series explores basic concepts of RAG. It is a classic application of context enrichment through similarity search in a vector store to generate contextualized answers based on document information.

This project uses embedding models and LLMs via Azure OpenAI. However, you can easily adapt the `AzureOpenaiService` class, from `azure_openai.py`, to use any other models.
```python
└── 📁src
    └── 📁services
        ├── __init__.py
        ├── azure_openai.py
```

```python
streamlit run projects/rag_faiss/app.py
```