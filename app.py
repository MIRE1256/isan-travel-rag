import os
import glob
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------- Configuration & Page Setup -----------------
st.set_page_config(
    page_title="ISAN Travel GuideBot - ไกด์ท่องเที่ยว 20 จังหวัดภาคอีสาน",
    page_icon="🧭",
    layout="centered"
)

# Custom CSS ตกแต่งให้สวยงาม ดูสบายตา สไตล์มินิมอล
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* กล่องหัวเรื่องหลัก */
    .hero-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.2rem 1.8rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #e0e7ff;
        font-weight: 300;
        line-height: 1.5;
    }
    
    /* กล่องปุ่มลัด */
    .quick-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #475569;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        color: #334155;
        font-size: 0.85rem;
        padding: 0.45rem 0.6rem;
        transition: all 0.2s ease;
        text-align: left;
    }
    .stButton>button:hover {
        border-color: #2563eb;
        background-color: #eff6ff;
        color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# แบนเนอร์หัวเว็บ
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧭 ไกด์ท่องเที่ยว 20 จังหวัดภาคอีสาน</div>
    <div class="hero-subtitle">ผู้ช่วย AI อัจฉริยะ แนะนำพิกัดเที่ยว เวลาเปิด-ปิด ร้านอาหารเด็ด และของฝากประจำถิ่น</div>
</div>
""", unsafe_allow_html=True)

# ----------------- Groq Client Setup -----------------
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ ไม่พบ GROQ_API_KEY ในระบบ กรุณาตรวจสอบการตั้งค่า")
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
    st.warning("⚠️ ไม่พบเอกสารข้อมูลในโฟลเดอร์ data/")
    st.stop()

# ----------------- Search Function -----------------
def search_context(query: str, top_k: int = 5, threshold: float = 0.28):
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

# ----------------- Chat Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "สวัสดีครับ! ยินดีต้อนรับสู่แดนอีสานบ้านเฮา 🌾 ผมคือ AI ไกด์นำเที่ยวประจำ 20 จังหวัดภาคอีสานครับ อยากไปเที่ยวจังหวัดไหน ถามพิกัด เวลาเปิด-ปิด ร้านอาหารเด็ด หรือของฝากขึ้นชื่อ พิมพ์ถามหรือกดเลือกคำถามแนะนำด้านล่างได้เลยครับ!"
        }
    ]

# ----------------- ปุ่มคำถามแนะนำยอดนิยม (10 คำถาม) -----------------
st.markdown('<div class="quick-title">💡 คำถามแนะนำยอดนิยม (คลิกเพื่อถามทันที):</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
sample_questions = [
    "อนุสาวรีย์ท้าวสุรนารี (ย่าโม) โคราช ตั้งอยู่ที่ไหนและเปิดเวลาใด",
    "ร้านไข่กระทะขึ้นชื่อในอุดรธานีเปิดกี่โมง",
    "ไปบุรีรัมย์ ซื้อของฝากอะไรดี",
    "เวลาเปิด-ปิด ผาแต้ม อุบลราชธานี",
    "วัดภูทอก บึงกาฬ เปิดกี่โมงและปิดช่วงไหน",
    "ของฝากขึ้นชื่อของยโสธรมีอะไรบ้าง",
    "สกายวอล์ค วัดผาตากเสื้อ หนองคาย อยู่ที่ไหน",
    "ร้านขนมจีนประโดก ครูยอด โคราช อยู่ที่ไหน",
    "มีทัวร์ดำน้ำดูปะการังที่กาฬสินธุ์ไหม",
    "กระเช้าลอยฟ้าขึ้นยอดภูเขาไฟร้อยเอ็ดราคาเท่าไหร่"
]

selected_query = None
for idx, q in enumerate(sample_questions):
    target_col = col1 if idx % 2 == 0 else col2
    if target_col.button(f"📌 {q}", key=f"btn_{idx}"):
        selected_query = q

# ----------------- Render ประวัติการแชท (ไม่โชว์ Sources) -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- รับ Input จาก Chat หรือ ปุ่มกด -----------------
user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับที่เที่ยวอีสานที่นี่...")
active_query = selected_query if selected_query else user_input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    retrieved_docs = search_context(active_query, top_k=5)

    if not retrieved_docs:
        context_str = "ไม่มีข้อมูลในคลังเอกสารที่เกี่ยวข้อง"
    else:
        context_blocks = [
            f"[{doc['source']}]: {doc['content']}"
            for doc in retrieved_docs
        ]
        context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "คุณคือ AI ไกด์นำเที่ยวผู้เชี่ยวชาญ 20 จังหวัดภาคอีสาน อัธยาศัยดี สุภาพ เป็นกันเอง\n"
        "แนวทางการตอบ:\n"
        "1. ทักทายทั่วไป/พูดคุย: หากผู้ใช้ทักทาย เช่น 'สวัสดี', 'มีที่ไหนแนะนำบ้าง' ให้ตอบรับอย่างเป็นมิตร แนะนำตัวเองว่าเป็นไกด์นำเที่ยว 20 จังหวัดอีสาน และเชิญชวนให้บอกชื่อจังหวัดที่สนใจหรือสไตล์การเที่ยว\n"
        "2. คำถามเจาะจงสถานที่/ร้าน/ของฝาก: ให้ตอบโดยยึดตาม 'บริบทที่ค้นพบ' (Context) เท่านั้น บอกพิกัดและเวลาเปิด-ปิดให้ชัดเจน เรียบเรียงให้อ่านง่าย สบายตา\n"
        "3. สิ่งที่ไม่มีจริงในภาคอีสาน หรือไม่มีในบริบท: ให้ตอบสุภาพว่า 'ขออภัยครับ ไม่พบข้อมูลดังกล่าวในเอกสารท่องเที่ยว' ห้ามกุข้อมูลขึ้นมาเด็ดขาด"
    )

    user_query_payload = f"บริบทที่ค้นพบ (Context):\n{context_str}\n\nคำถามของผู้ใช้: {active_query}"

    with st.chat_message("assistant"):
        with st.spinner("ไกด์กำลังค้นหาข้อมูลให้ครับ..."):
            try:
                available_models = [
                    m.id for m in groq_client.models.list().data 
                    if "whisper" not in m.id and "guard" not in m.id and "vision" not in m.id
                ]
                priority_list = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "qwen/qwen3.8-27b",
                    "mixtral-8x7b-32768"
                ]
                selected_model = next((m for m in priority_list if m in available_models), available_models[0] if available_models else None)

                response = groq_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query_payload}
                    ],
                    temperature=0.2,
                    max_tokens=650
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_reply
                })

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Groq API: {str(e)}")

    # จุดมาร์กเกอร์ด้านล่างสุด และสั่ง Scroll ลงมาทันทีที่มีการถามคำถาม
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
    components.html(
        """
        <script>
            setTimeout(function() {
                var el = window.parent.document.getElementById('chat-bottom');
                if (el) {
                    el.scrollIntoView({behavior: 'smooth', block: 'end'});
                }
            }, 300);
        </script>
        """,
        height=0
    )