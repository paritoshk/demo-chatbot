import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from file import UploadedFile

prompt_template = """\
You are Prometheus, a bot who helps VCs in analyzing startups.
Your task is to answer questions about companies using there pitchdeck.
If you dont know the answer to any question, ask the user to clarify or apologize

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

load_dotenv()


async def main():
    async def conversational_chat(query):
        print(query)
        result = qa({"question": query}, return_only_outputs=True)
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Creating the chatbot interface
    st.title("VCopilot Testing :")

    if "ready" not in st.session_state:
        st.session_state["ready"] = False

    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Add your code here that needs to be executed
            uploaded_file.seek(0)
            file = UploadedFile(uploaded_file)
            vectors = file.get_vector()
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-4"),
                retriever=vectors.as_retriever(),
                return_source_documents=False,
                memory=memory,
                qa_prompt=PROMPT,
            )

        st.session_state["ready"] = True

    st.divider()

    if st.session_state["ready"]:
        if "generated" not in st.session_state:
            st.session_state["generated"] = [
                "Welcome! You can now ask any questions regarding the uploaded pitchdeck"
            ]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Query:",
                    placeholder="Ask information about pitchdeck",
                    key="input",
                )
                submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

        if st.session_state["generated"]:
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    message(
                        st.session_state["past"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )
                    message(
                        st.session_state["generated"][i],
                        key=str(i),
                    )


if __name__ == "__main__":
    asyncio.run(main())
