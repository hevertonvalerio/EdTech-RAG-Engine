import streamlit as st
import os
import tempfile

from query_db import search_knowledge_base, generate_response
from indexer import index_pdf

# Page config
st.set_page_config(page_title="RAG EdTech Engine", layout="centered")

#App title
st.title("RAG EdTech Egine")
st.subheader("Dinamic generator of Content and Quizzes")

tab1, tab2 = st.tabs(["Upload PDF", "Ask Questions"])

with tab1:
    st.write("Load a PDF to index into the knowledge base.")

    uploaded_file = st.file_uploader("Choose a PDF", type = "pdf")

    if uploaded_file is not None:
        # Show loading spinner
        with st.spinner("Processing and indexing the PDF..."):
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name 

            try:
                # index the PDF
                num_chuncks =index_pdf(tmp_path)
                st.success(f"PDF indexed with {num_chuncks} chunks")
            except Exception as e:
                st.error(f"Error indexing PDF: {e}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

with tab2:
    st.write("Make questions to the knowledge base.")

    #Inputs
    topic = st.text_input(
        "Type the subject wanted",
        placeholder="Ex: a transição do pensamento concreto para o formal na robótica"
    )

    #Buttons
    if st.button("Generate Quiz/Explanation", type="primary"):
        if topic.strip():
            #using st.spinner to show a loading animation
            with st.spinner("Consulting vector database and generating response..."):
                try:
                    # 1. Define the topic
                    question = f"Crie um quiz de múltipla escolha sobre: {topic}"
                    
                    # 2. Search on knowledge base and save the results on variable docs
                    docs = search_knowledge_base(topic)
                    
                    # 3. Generate the response with the documents founded on vector database
                    response = generate_response(question, docs)
                    
                    # 4. Print the response
                    st.success("Pronto!")
                    st.markdown("### Resultado")
                    st.write(response)
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar a requisição: {e}")
        else:
            st.warning("Por favor, digite um tópico antes de clicar no botão.")