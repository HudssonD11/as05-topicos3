import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
# import handling_pdfs
import fitz
import faiss
import os


model_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
os.makedirs("arquivos", exist_ok=True)




st.title("AS05 - Assistente Conversacional Baseado em LLM")

API_KEY = 'AIzaSyA7IywZsH4XRjUopxTLpG7jmqPAQoLzyHI'
model = genai.GenerativeModel('gemini-1.5-flash-latest')
genai.configure(api_key=API_KEY)

def ajustar_pergunta(pergunta, arquivos):
    if arquivos:
        textos_documentos = []
        texto = ""
        for arquivo in arquivos:
            pdf_file = os.path.join("arquivos", arquivo.name)
            pdf_document = fitz.open(pdf_file)
            for pagina in pdf_document:
                texto_tmp = pagina.get_text()
                textos_documentos.append(texto_tmp)
                texto += texto_tmp

        sentences = texto.split('\n')
        embeddings = model_embeddings.encode(sentences)

        dim = embeddings.shape[1]
        indice = faiss.IndexFlatL2(dim)
        indice.add(embeddings)

        question_embedding = model_embeddings.encode([pergunta])
        distancias, indices = indice.search(question_embedding, k=5)
        respostas_documentos = []
        for i in indices[0]:
            respostas_documentos.append(sentences[i])
            for j in range(i - 5, i + 5 + 1):
                    if 0 <= j < len(sentences):
                        respostas_documentos.append(sentences[j])        
        nova_pergunta = "Responda essa pergunta dados os contextos que serão apresentados abaixo (responda apenas com a resposta da pergunta como se estivesse conversando com uma pessoa): Pergunta: " + pergunta + "\n\nContexto: ".join(respostas_documentos)
        print(nova_pergunta)
        return nova_pergunta

arquivos = st.file_uploader("Selecionar arquivos", accept_multiple_files=True, type=["pdf"])
if arquivos:
    for arquivo in arquivos:
        with open(os.path.join("arquivos", arquivo.name), "wb") as f:
            f.write(arquivo.getbuffer())

pergunta = st.chat_input("Fale algo e será respondido com base nos arquivos selecionados: ")
if pergunta is not None:
    if arquivos:
        pergunta = ajustar_pergunta(pergunta, arquivos)
    with st.spinner("Por favor, aguarde enquanto a resposta é gerada..."):
        try:
            stream = model.generate_content(pergunta)
            st.write(stream.text)
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")
