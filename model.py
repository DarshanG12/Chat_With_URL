from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

template = """
    Use the following context (delimited by <ctx></ctx>)  to answer the question:
    You provide information based on the stored context  on the database for answers..
    If a question is outside the stored context, you should reply with 'I don't have information on that topic.' 
    However, it is crucial to remember that you should only answer based on the information stored in documents.
    {context}
    </ctx>
    ------
    {question}
    Answer:
    """

prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt},
        verbose=True
    )
    return conversation_chain
