import openai
import streamlit as st
from streamlit_chat import message
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Pinecone
import pinecone
from tqdm.autonotebook import tqdm
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ['OPENAI_API_KEY'] = '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'
OPENAI_API_KEY="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"

# Setting page title and header
st.set_page_config(page_title="Tanya HRD", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Tanya HRD kamu </h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("MENU")
#model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
model_name = "GPT-3.5"
source_file = st.sidebar.radio("Anda mau tanya tentang apa? ", ("Peraturan perusahaan", "Pelatihan Vave"))
#counter_placeholder = st.sidebar.empty()
#counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# Map source file according to radio button
if source_file =="Peraturan perusahaan":
    source = "1m1IEUGbdpTeap4Ur85XXNVQRwR-5Wcu2"
else:
    source = "1bJFdl5FVdbhIvH39IkduKB3JaNg1fNIX"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    #counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")



def generate_from_file(prompt):
    st.session_state['messages'].append({"role":"user","content": prompt})

    loader = GoogleDriveLoader(folder_id=source,credentials_path="D:\Belajar\LearnLang\credentials.json" )
    print(loader)
    documents =loader.load()
    print(documents)
    print(len(documents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_texts = text_splitter.split_documents(documents)

    print(len(doc_texts))

    embedings = OpenAIEmbeddings(openai_api_key='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')

    pinecone.init(api_key="10144012-e2f8-4942-a114-ee8854e1996e",
              environment="us-east-1-aws")

    index_name = "example-index-2"

    docsearch = Pinecone.from_texts([t.page_content for t in doc_texts], embedings, index_name=index_name)

    #query = input("apa yang mau anda tanyakan? ")

    docs = docsearch.similarity_search(prompt, include_metadata=True)

    llm = OpenAI(temperature=0, openai_api_key="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")
    chain = load_qa_chain(llm,chain_type="stuff")

    response = chain.run(input_documents=docs, question=prompt)

    total_tokens = "1"
    prompt_tokens = "1"
    completion_tokens = "1"

    return response, total_tokens, prompt_tokens, completion_tokens

response_container = st.container()
container = st.container()

with container:
    prompt = st.text_input("Apa yang mau anda tanyakan?",key="my-form",)
  
    #with 
    # st.form(key='my_form', clear_on_submit=True):
    #     user_input = st.text_area("You:", key='input', height=100)
    #     submit_button = st.form_submit_button(label='Send')

        
    #if submit_button and user_input:
    if prompt:
        #output, total_tokens, prompt_tokens, completion_tokens = generate_response(prompt)
        output, total_tokens, prompt_tokens, completion_tokens = generate_from_file(prompt)
        # st.session_state['past'].append(user_input)
        st.session_state['past'].append(prompt)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            #cost = total_tokens * 0.002 / 1000
            cost = 1
        else:
            #cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            cost = 1

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost





if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            #st.write(f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            #counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

