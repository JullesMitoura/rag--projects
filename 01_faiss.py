import os
import tempfile
import streamlit as st
from src.utils.settings import Settings
from src.utils.chat_template import ChatTemplate
from src.services.azure_openai import AzureOpenaiService
from src.utils.extractors import DocumentExtractor
from src.services.faiss import FaissService


@st.cache_resource
def load_services():
    sets = Settings()
    azai_serv = AzureOpenaiService(sets=sets)
    llm = azai_serv.get_llm()
    embeddings = azai_serv.get_embeddings()
    extractor = DocumentExtractor()
    faiss_service = FaissService(embeddings=embeddings,
                                   chunk_size=1200,
                                   chunk_overlap=500)
    return llm, embeddings, extractor, faiss_service


def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return temp_dir


llm, embeddings, extractor, faiss_service = load_services()


def invoke_llm(history, prompt):
    context = ""

    if "vdb_temp_id" in st.session_state:
        vdb_id = st.session_state["vdb_temp_id"]
        results = faiss_service.similarity_search(vdb_id, query=prompt, k=3)
        retrieved_text = "\n".join([doc.page_content for doc in results])
        context += f"Relevant documents:\n{retrieved_text}\n\n"

    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"

    context += f"User: {prompt}\nAssistant:"

    res = llm.invoke(context).content
    return res


st.set_page_config(
    page_title="RAG with FAISS",
    page_icon="📚",
    layout="wide"
)

def main():
    st.markdown("""
    <div style="margin-top: 1rem; margin-bottom: 0rem;">
        <h1>RAG with FAISS</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Document Upload")
        
        if "vdb_temp_id" in st.session_state:
            pass
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to create your knowledge base"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            for file in uploaded_files:
                st.write(f"📄 {file.name}")
        
        st.markdown("---")
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    temp_dir = save_uploaded_files(uploaded_files)
                    text = extractor.extract_documents(path=temp_dir)

                    # create temporary vDB
                    documents = faiss_service.text_to_documents(text)
                    temp_id, vdb = faiss_service.create_temporary_database(documents=documents)

                    st.session_state['vdb_temp_id'] = temp_id
                    st.session_state['vdb'] = vdb

                    st.success("Vector Database created successfully!")
            else:
                st.warning("Please upload documents first")

    with col2:
        st.header("Chat")
        chat_template = ChatTemplate(
            process_message=invoke_llm,
            placeholder="Talk to me..."
        )
        chat_template.chat()


if __name__ == "__main__":
    main()