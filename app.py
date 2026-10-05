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
    page_title="ไกด์นำเที่ยว 20 จังหวัดภาคอีสาน - ISAN AI Guide",
    page_icon="🌾",
    layout="centered"
)

# Custom CSS สไตล์มินิมอล โมเดิร์น สะอาดตา
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
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
    <div class="hero-title">🌾 ไกด์ท่องเที่ยว 20 จังหวัดภาคอีสาน</div>
    <div class="hero-subtitle">ผู้ช่วย AI อัจฉริยะ แนะนำพิกัดเที่ยว 12 แห่ง ร้านเด็ด 5 ร้าน ของฝาก และสิ่งที่ต้องไปเฮ็ดครบ 20 จังหวัด!</div>
</div>
""", unsafe_allow_html=True)

# ----------------- Groq Client Setup -----------------
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ ไม่พบ GROQ_API_KEY ในระบบ กรุณาตรวจสอบการตั้งค่า Secrets")
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

    for file_path in data_files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) < 320:
                current_chunk += "\n" + line if current_chunk else line
            else:
                if len(current_chunk.strip()) > 15:
                    chunks.append(current_chunk.strip())
                    chunk_metadata.append({"source": file_name, "snippet": current_chunk.strip()})
                current_chunk = line
        if len(current_chunk.strip()) > 15:
            chunks.append(current_chunk.strip())
            chunk_metadata.append({"source": file_name, "snippet": current_chunk.strip()})

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
    st.warning("⚠️ ไม่พบเอกสารข้อมูลในโฟลเดอร์ data/ กรุณาตรวจสอบไฟล์ .txt ของคุณ")
    st.stop()

# ----------------- Hybrid Search Function -----------------
ISAN_PROVINCES = [
    "นครราชสีมา", "โคราช", "ขอนแก่น", "อุดรธานี", "อุบลราชธานี", "นครพนม", "เลย",
    "หนองคาย", "ร้อยเอ็ด", "บุรีรัมย์", "บึงกาฬ", "สกลนคร", "สุรินทร์", "ศรีสะเกษ",
    "กาฬสินธุ์", "ชัยภูมิ", "มหาสารคาม", "มุกดาหาร", "ยโสธร", "หนองบัวลำภู", "อำนาจเจริญ"
]

def search_context(query: str, top_k: int = 5):
    results = []
    
    # 1. ค้นหาแบบตรงชื่อจังหวัดและประเภทข้อมูล (ที่เที่ยว, ร้านเด็ด, ของฝาก, สิ่งที่ต้องทำ)
    for meta in doc_meta:
        chunk_text = meta["snippet"]
        for prov in ISAN_PROVINCES:
            if prov in query:
                if prov in chunk_text or prov in meta["source"]:
                    if any(kw in query for kw in ["ของฝาก", "ของดี", "ซื้อ", "ของกินฝาก"]) and "ของฝาก" in chunk_text:
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})
                    elif any(kw in query for kw in ["ร้านอาหาร", "ของกิน", "อาหาร", "กินข้าว", "แซ่บ"]) and "ร้านอาหาร" in chunk_text:
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})
                    elif any(kw in query for kw in ["ต้องทำ", "ห้ามพลาด", "ต้องไปทำ", "ไฮไลท์"]) and "สิ่งที่เมื่อมาจังหวัดนี้แล้วต้องทำ" in chunk_text:
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})
                    elif any(kw in query for kw in ["เปิด", "ปิด", "เวลา", "ตั้งอยู่", "พิกัด", "ที่เที่ยว", "สถานที่"]) and "สถานที่ท่องเที่ยว" in chunk_text:
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})

    # 2. Vector Semantic Search
    model = load_embedding_model()
    q_vec = model.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, top_k)
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1:
            snippet = doc_meta[idx]["snippet"]
            if snippet not in [r["content"] for r in results]:
                results.append({
                    "score": float(score),
                    "source": doc_meta[idx]["source"],
                    "content": snippet
                })
    return results[:6]

# ----------------- Chat Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "สบายดีครับพี่น้อง! ยินดีต้อนรับสู่แดนอีสานบ้านเฮาเด้อครับ 🌾 ข้อยคือ AI ไกด์นำเที่ยว 20 จังหวัดภาคอีสาน อยากไปเลาะไส ถามพิกัด ที่เที่ยว เวลาเปิด-ปิด ร้านอาหารแซ่บๆ ของฝาก หรือกิจกรรมเด็ดที่ต้องไปเฮ็ด พิมพ์ถามหรือจิ้มปุ่มคำถามแนะนำด้านบนได้เลยเด้อครับ!"
        }
    ]

# ----------------- ปุ่มคำถามแนะนำยอดนิยม 12 ข้อ -----------------
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
    "ไปเที่ยวเลย มีอะไรที่ต้องทำหรือห้ามพลาดบ้าง",
    "มาขอนแก่น ต้องไปทำกิจกรรมอะไรบ้างถึงจะเรียกว่ามาถึง",
    "ร้านสเต็กเนื้อโคขุนโพนยางคำ สกลนคร ร้านไหนเด็ด",
    "มีทัวร์ดำน้ำดูปะการังที่กาฬสินธุ์ไหม",
    "กระเช้าลอยฟ้าขึ้นยอดภูเขาไฟร้อยเอ็ดราคาเท่าไหร่"
]

selected_query = None
for idx, q in enumerate(sample_questions):
    target_col = col1 if idx % 2 == 0 else col2
    if target_col.button(f"📌 {q}", key=f"btn_{idx}"):
        selected_query = q

# ----------------- Render ประวัติการแชท (ไม่แสดง Sources) -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- รับ Input จาก Chat หรือ ปุ่มกด -----------------
user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับที่เที่ยวอีสานที่นี่เด้อ...")
active_query = selected_query if selected_query else user_input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    retrieved_docs = search_context(active_query, top_k=5)

    context_blocks = [
        f"[{doc['source']}]:\n{doc['content']}"
        for doc in retrieved_docs
    ]
    context_str = "\n\n".join(context_blocks)

    # ปรับ System Prompt ให้เป็น AI ไกด์อีสาน อารมณ์ดี เว้าม่วน
    system_prompt = (
        "คุณคือ 'AI ไกด์นำเที่ยวอีสานบ้านเฮา' ผู้เชี่ยวชาญการท่องเที่ยว 20 จังหวัดภาคอีสานของประเทศไทย\n"
        "บุคลิกและสไตล์การพูด:\n"
        "- ร่าเริง อารมณ์ดี สุภาพ เป็นกันเอง มีสำเนียงและคำสร้อยแบบคนอีสาน เช่น 'เด้อครับ', 'น้อครับ', 'แซ่บหลาย', 'อีหลี', 'ไปเลาะ', 'เฮ็ด'\n"
        "- ตอบเป็นภาษาไทยมาตรฐานที่แทรกคำอีสานอย่างน่ารัก อ่านเข้าใจง่าย สบายตา\n\n"
        "แนวทางการตอบ:\n"
        "1. ทักทาย/คุยทั่วไป: หากผู้ใช้ทักทาย ให้ตอบรับอย่างอบอุ่นแบบคนอีสาน ยินดีต้อนรับสู่แดนอีสานบ้านเฮา และเชิญชวนไปเลาะเที่ยว\n"
        "2. คำถามสถานที่ ร้านอาหาร ของฝาก สิ่งที่ต้องไปทำ: ให้ตอบโดยยึดตาม 'ข้อมูลอ้างอิง' (Context) ที่ค้นพบเท่านั้น บอกชื่อ พิกัด เวลาเปิด-ปิด และกิจกรรมเด็ดให้ชัดเจน ครบถ้วน\n"
        "3. สำหรับสถานที่หรือกิจกรรมที่ไม่มีจริงในภาคอีสาน หรือไม่มีในข้อมูลอ้างอิง (เช่น ดำน้ำดูปะการัง, กระเช้าลอยฟ้าข้ามภูเขาไฟ): ให้ตอบปฏิเสธอย่างสุภาพปนน่ารักว่า 'ขออภัยเด้อครับ ข้อยบ่พบข้อมูลนี้ในเอกสารท่องเที่ยว' และอธิบายสั้นๆ ว่าในจังหวัดนั้นไม่มีกิจกรรมดังกล่าวอีหลีครับ"
    )

    user_query_payload = f"ข้อมูลอ้างอิง (Context):\n{context_str}\n\nคำถาม: {active_query}"

    with st.chat_message("assistant"):
        with st.spinner("ไกด์กำลังฟ้าวหาข้อมูลให้เด้อครับ..."):
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
                    temperature=0.25,
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

    # Auto-Scroll ลงมาที่คำตอบล่าสุด
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