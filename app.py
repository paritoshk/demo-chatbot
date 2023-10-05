import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from file import UploadedFile
# List of questions for the chatbot
questions = {
  "Target Market": [
    "Who's the customer? Key characteristics?",
    "Is the market large now or in the future?",
    "What is the growth rate?",
    "How do you calculate the market size: # potential customers x average revenue per customer?",
    "Is the Serviceable Available Market (SAM) $10B+?",
    "What is the Serviceable Obtainable Market (SOM)?"
  ],
  "Problem or Need": [
    "What's the core problem/need to be solved?",
    "How severe is the pain or challenge?"
  ],
  "Solution": [
    "What is the core value proposition?",
    "Does it provide better, faster, and/or cheaper solutions?",
    "What is the ROI for customers?",
    "What is the magnitude of improvement over the status quo?",
    "What is the unique approach?",
    "Is there a strong brand perspective?"
  ],
  "Team, Board, Advisors": [
    "What is the team's prior career?",
    "What are their successes/failures?",
    "Do they have industry knowledge?",
    "What are they uniquely skilled at: raising capital, generating revenues, building a team, exiting?",
    "Do they have any key relationships?",
    "What is the secret they unlocked?"
  ],
  "Traction": [
    "Has a Minimum Viable Product been built?",
    "What are the key things learned?",
    "Has the ROI been validated?",
    "What are the key metrics: # Customers, ARR (Annual Recurring Revenue)?",
    "Is there a revenue pipeline?",
    "Are there any strategic partnerships?",
    "What is the capital efficiency: ARR / Capital Spent To Scale?",
    "What is the LTV (Lifetime Value) / CAC (Customer Acquisition Cost)?"
  ],
  "Revenue Model": [
    "What is the revenue per customer?",
    "Is it a subscription-based or one-time revenue model?",
    "What is the frequency of revenue generation?",
    "How is the LTV calculated?",
    "What are the ways to increase LTV?",
    "What is the length of the sales cycle?",
    "Is it a high-price or high-volume strategy?"
  ],
  "Strategy": [
    "What are the key expenses/time efforts involved?",
    "What are the COGs (Cost of Goods Sold)?",
    "What is the gross margin percentage?",
    "How can the gross margin be improved?",
    "What are the biggest expenses?",
    "Where is the majority of time spent?",
    "What is the Cost to Acquire & Maintain Customers (CAC)?",
    "What is the marketing channel strategy?"
  ],
  "Financials": [
    "What were the last 1-2 year revenues?",
    "What are the projected 1-5 year revenues?",
    "What is the breakdown of gross vs. net revenue?",
    "What are the burn rate per month and cash runway?",
    "What is the path to profitability?",
    "How much total capital has been spent: % Build vs. % Scale?",
    "What is the capital efficiency: ARR / Capital Spent To Scale?",
    "Is the goal to maintain or improve capital efficiency?",
    "What market penetration percentage is required to hit revenue targets?"
  ],
  "Competition": [
    "Who are the direct and indirect competitors?",
    "What are the barriers to entry?",
    "What is the main differentiation from competitors?",
    "Are there simpler alternatives that are good enough?",
    "Is there a risk of future obsolescence?",
    "Is there an unfair and/or sustainable competitive advantage?",
    "Are there any patents or key partnerships?",
    "What are the key risks (current and future)?"
  ],
  "Exit Opportunity": [
    "What types of acquirers are targeted?",
    "Why buy instead of building a similar solution?",
    "Is there potential for an IPO?",
    "What revenue multiple is expected?",
    "Is there a possibility of achieving a 10x+ return with dilution?"
  ],
  "Investment Terms": [
    "How much funding is being raised?",
    "What is the amount committed vs. available?",
    "What is the pre-money and post-money valuation?"
  ],
  "Post-$ Valuation / ARR": [
    "Who are the new and previous investors?",
    "Are previous investors coming back?",
    "What is the total capital previously raised?",
    "Is the preferred investment senior or pari-passu?",
    "Are there any debts/SAFEs/notes?"
  ],
  "Strategic Value": [
    "How can potential partners help?",
    "Can they provide introductions to customers, partners, strategics, investors, or employees?"
  ]
}
prompt_template = """\
You are Theus AI, an helpful assistant who helps venture investors in analyzing startups specifically on Competitors, Market Sizing and Team.
Your task is to answer questions about companies using there pitchdeck. 
"When asked a question" - You must use detailed math in step by step market sizing calculations, and give competitors (if you cannot find any around the stage you can give examples (creative and hypothetical).
If you dont know the answer to any question, ask the user to clarify or apologize.
Use as much creative information in your database to answer the question before apologizing, or make an hypothesis and disclaim it.   

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
    st.title("VentureCopilot Demo :")

    if "ready" not in st.session_state:
        st.session_state["ready"] = False

    uploaded_file = st.file_uploader("Choose a file", type="pdf",accept_multiple_files=True)

    if uploaded_file is not None:
        with st.spinner("Processing..."):
# The code you mentioned is responsible for processing the uploaded file. It reads the file, creates
# an instance of the `UploadedFile` class, and then generates vectors from the file using the
# `get_vector()` method. These vectors are used for retrieval in the conversational chat.
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
