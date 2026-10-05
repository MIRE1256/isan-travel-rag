
import os
import glob
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------- Configuration & Page Setup -----------------
st.set_page_config(
    page_title="คู่มือท่องเที่ยว 20 จังหวัดภาคอีสาน - RAG GuideBot",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 ไกด์ท่องเที่ยว 20 จังหวัดภาคอีสาน (All Isan 20 Provinces AI Guide)")
st.caption("ระบบถาม-ตอบสถานที่ท่องเที่ยว เวลาเปิด-ปิด พิกัดที่ตั้ง ร้านอาหารเด็ด และของฝากครบ 20 จังหวัดภาคอีสาน ด้วยเทคนิค RAG")

# อ่าน API Key จาก secrets ของ Streamlit หรือจาก Environment
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ ไม่พบ GROQ_API_KEY ใน Streamlit Secrets กรุณาตั้งค่าก่อนใช้งาน")
    st.stop()

groq_client = Groq(api_key=api_key)

# ----------------- Model & Vector Caching -----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def build_vector_store():
    model = load_embedding_model()
    data_files = sorted(glob.glob("data/*.txt"))
    
    chunks = []
    chunk_metadata = []
    chunk_size = 400
    overlap = 80

    for file_path in data_files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned_text = " ".join(text.split())
        for i in range(0, len(cleaned_text), chunk_size - overlap):
            chunk = cleaned_text[i : i + chunk_size]
            if len(chunk.strip()) > 30:
                chunks.append(chunk)
                chunk_metadata.append({"source": file_name, "snippet": chunk})

    if not chunks:
        return None, [], []

    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, chunks, chunk_metadata

index, doc_chunks, doc_meta = build_vector_store()

if index is None or len(doc_chunks) == 0:
    st.warning("⚠️ ไม่พบเอกสารในโฟลเดอร์ data/ กรุณาตรวจสอบไฟล์ .txt ของคุณ")
    st.stop()

# ----------------- Retrieval Function -----------------
def search_context(query: str, top_k: int = 5, threshold: float = 0.30):
    model = load_embedding_model()
    q_vec = model.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, top_k)
    results = []

    for score, idx in zip(distances[0], indices[0]):
        if idx != -1 and score >= threshold:
            results.append({
                "score": float(score),
                "source": doc_meta[idx]["source"],
                "content": doc_meta[idx]["snippet"]
            })
    return results

# ----------------- Chat Interface -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "สวัสดีครับ! ผมคือ AI ไกด์นำเที่ยว 20 จังหวัดภาคอีสาน สามารถสอบถามข้อมูลที่เที่ยว เวลาเปิด-ปิด พิกัดที่ตั้ง ร้านอาหารเด็ด และของฝากประจำจังหวัดได้เลยครับ",
            "sources": []
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 เอกสารอ้างอิง (Context Sources)"):
                for src in msg["sources"]:
                    st.write(f"- **{src['source']}** (Relevance: {src['score']:.2f})")
                    st.caption(f"_{src['content']}_")

user_prompt = st.chat_input("พิมพ์คำถามท่องเที่ยว 20 จังหวัดอีสานที่นี่...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    retrieved_docs = search_context(user_prompt, top_k=5)

    if not retrieved_docs:
        context_str = "ไม่มีข้อมูลในคลังเอกสารที่เกี่ยวข้อง"
    else:
        context_blocks = [
            f"[แหล่งข้อมูล: {doc['source']}]\n{doc['content']}"
            for doc in retrieved_docs
        ]
        context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "คุณคือ AI ไกด์ท่องเที่ยวผู้เชี่ยวชาญข้อมูล 20 จังหวัดภาคอีสานของประเทศไทย\n"
        "กฎเหล็กในการตอบคำถาม:\n"
        "1. ตอบคำถามโดยยึดตาม 'บริบทที่ค้นพบ' (Context) ด้านล่างนี้เท่านั้น\n"
        "2. หากมีข้อมูล ให้บอกเวลาทำการ พิกัดที่ตั้ง หรือร้านอาหาร/ของฝากตามที่ระบุในบริบทอย่างชัดเจน\n"
        "3. หากในบริบทไม่มีคำตอบ หรือข้อมูลไม่เพียงพอ คุณต้องตอบปฏิเสธว่า 'ขออภัยครับ ไม่พบข้อมูลดังกล่าวในเอกสารท่องเที่ยว' ห้ามกุหรือคาดเดาข้อมูลขึ้นมาเองโดยเด็ดขาด\n"
        "4. ระบุชื่อไฟล์แหล่งอ้างอิงท้ายคำตอบเสมอ เช่น [อ้างอิง: filename.txt]"
    )

    user_query_payload = f"บริบทที่ค้นพบ (Context):\n{context_str}\n\nคำถาม: {user_prompt}"

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลจากคลังท่องเที่ยว 20 จังหวัดอีสาน..."):
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query_payload}
                    ],
                    temperature=0.1,
                    max_tokens=650
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)

                if retrieved_docs:
                    with st.expander("📚 เอกสารอ้างอิง (Context Sources)"):
                        for src in retrieved_docs:
                            st.write(f"- **{src['source']}** (Relevance: {src['score']:.2f})")
                            st.caption(f"_{src['content']}_")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_reply,
                    "sources": retrieved_docs
                })

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Groq API: {str(e)}")