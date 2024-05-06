import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate


def get_url_text(urls):
    text = ""
    for url in urls:
        if url.strip():  # Check if URL is not empty
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text += soup.get_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Define the template
template = """
Use the following context (delimited by <ctx></ctx>)  to answer the question:
You provide information based on the stored context  on the database for answers. 
If a question is outside the stored context, you should reply with 'I don't have information on that topic.' 
However, it is crucial to remember that you should only answer based on the information stored in documents.
{context}
</ctx>
------
<hs>
{chat_history}
</hs>
------
{question}
Answer:
"""

# Create the prompt
prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={'prompt': prompt},
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with URLs",
                       page_icon=":globe:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with URLs :globe_with_meridians:")
    user_question = st.text_input("Ask a question about the URLs:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your URLs")
        urls = st.text_area("Enter URLs")
        urls = urls.split('\n')
        if st.button("Process"):
            with st.spinner("Processing"):
                # get text from URLs
                raw_text = get_url_text(urls)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

