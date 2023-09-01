from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Add your Next.js frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file.file)
    for page in pdf_reader.pages:
        text += page.extract_text()
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
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



@app.post('/recieve-pdf')
def recieve(pdf_file: UploadFile = File(...)):
    load_dotenv()
    try:
        raw_text = get_pdf_text(pdf_file)
        
        text_chunks = get_text_chunks(raw_text)
        
        vectorstore = get_vectorstore(text_chunks)

        global conversation
        
        conversation = get_conversation_chain(vectorstore)


        vectorstore_info = {
            "num_documents": len(text_chunks),
        
            # Add more relevant information here
        }
        return JSONResponse(content={"vectorstore_info": vectorstore_info}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/ask/{user_question}")
def handle_userinput(user_question: str):
    try:
        global conversation
        
        if 'conversation' not in globals():
            return JSONResponse(content={"answer": "Please upload a PDF and initialize the conversation chain first."}, status_code=400)
        print(user_question)
        res = conversation({'question':user_question})
        # print(res)
        ai_response = res['chat_history'][1].content
        return JSONResponse(content={"answer": ai_response}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)  




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
