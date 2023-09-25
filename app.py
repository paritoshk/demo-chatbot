from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
from langchain.document_loaders import TextLoader

import ocr

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  

async def main():
    async def storeDocEmbeds(uploaded_file):
        
        with open(f"files/pd/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        images_folder = ocr.create_images_from_pdf(f"files/pd/{uploaded_file.name}")
        text_file = ocr.extract_text_from_image(images_folder)
        loader = TextLoader(text_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(documents, embeddings)
  
        with open(f"files/pk/{uploaded_file.name}.pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds(uploaded_file):
        
        if not os.path.isfile(f"files/pk/{uploaded_file.name}.pkl"):
            await storeDocEmbeds(uploaded_file)
        
        with open(f"files/pk/{uploaded_file.name}.pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []


    #Creating the chatbot interface
    st.title("VCopilot Testing :")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    if uploaded_file is not None:

        with st.spinner("Processing..."):
        # Add your code here that needs to be executed
            uploaded_file.seek(0)
            vectors = await getDocEmbeds(uploaded_file)
            qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-4"), retriever=vectors.as_retriever(), return_source_documents=True)

        st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: Summarize the paper in a few sentences", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


if __name__ == "__main__":
    asyncio.run(main())