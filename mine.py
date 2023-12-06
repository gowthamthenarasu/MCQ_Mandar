import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os 


load_dotenv()
# Set API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# PDF processing
pdfreader = PdfReader('XYZ_contract_pdf_Sumit Yenugwar.pdf')
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Text splitting
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()


# Create document search
document_search = FAISS.from_texts(texts, embeddings)

#########################################################
document_search.save_local("faiss_index")

#below lines loads the vectorized data that was saved in previous code line
new_document_search = FAISS.load_local("faiss_index", embeddings)
 

##################################

# Load QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

with st.sidebar:
    st.title('🤗💬 LLM Chat APP')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space
    st.write('Made with ❤️ by [Prompt Engineer](https://www.youtube.com/watch?v=M4mc-z_K1NU&list=PLUTApKyNO6MwrOioHGaFCeXtZpchoGv6W)')

# Streamlit app
def main():
    st.title("DAMA-Data Management body of knowledge")

    # Text input area
    user_input = st.text_area("Enter your MCQ question ",height=150)

    # Button to trigger model inference
    if st.button("Get Answer"):
        # Combine user input with the prompt and query
        prompt_query = f"you have provided with MCQ question and its option as a chatbot model: {user_input}"
        text_query = prompt_query + user_input

        # Perform similarity search
        docs = new_document_search.similarity_search(text_query)

        # Run the model with the combined text and query
        model_answer = chain.run(input_documents=docs, question=user_input)

        # Display the model's answer
        st.text_area("Model Answer:", value=model_answer)

# Run the Streamlit app
if __name__ == "__main__":
    main()
