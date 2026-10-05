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
    page_icon="🌾",
    layout="centered"
)

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

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌾 ไกด์ท่องเที่ยว 20 จังหวัดภาคอีสาน</div>
    <div class="hero-subtitle">พาเลาะ 20 จังหวัดอีสานบ้านเฮา ครบทั้งพิกัด ร้านแซ่บ ของฝาก และไฮไลท์ห้ามพลาด คัดสรร 12 ที่เที่ยว 5 ร้านเด็ด และกิจกรรมห้ามพลาด จัดเต็มทุกสาย</div>
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
            if len(current_chunk) + len(line) < 350:
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
    st.warning("⚠️ ไม่พบเอกสารข้อมูลในโฟลเดอร์ data/ กรุณาตรวจสอบไฟล์ .txt")
    st.stop()

# ----------------- Search Function -----------------
PROVINCE_MAP = {
    "โคราช": "นครราชสีมา",
    "นครราชสีมา": "นครราชสีมา",
    "ขอนแก่น": "ขอนแก่น",
    "อุดร": "อุดรธานี",
    "อุดรธานี": "อุดรธานี",
    "อุบล": "อุบลราชธานี",
    "อุบลราชธานี": "อุบลราชธานี",
    "นครพนม": "นครพนม",
    "เลย": "เลย",
    "หนองคาย": "หนองคาย",
    "ร้อยเอ็ด": "ร้อยเอ็ด",
    "บุรีรัมย์": "บุรีรัมย์",
    "บึงกาฬ": "บึงกาฬ",
    "สกลนคร": "สกลนคร",
    "สุรินทร์": "สุรินทร์",
    "ศรีสะเกษ": "ศรีสะเกษ",
    "กาฬสินธุ์": "กาฬสินธุ์",
    "ชัยภูมิ": "ชัยภูมิ",
    "สารคาม": "มหาสารคาม",
    "มหาสารคาม": "มหาสารคาม",
    "มุกดาหาร": "มุกดาหาร",
    "ยโสธร": "ยโสธร",
    "หนองบัวลำภู": "หนองบัวลำภู",
    "อำนาจเจริญ": "อำนาจเจริญ"
}

def search_context(query: str, top_k: int = 6):
    results = []
    
    # 1. เช็คว่ามีชื่อจังหวัดในคำถามหรือไม่ ถ้ามี ให้ดึงข้อมูลทั้งหมดของจังหวัดนั้นมาใส่ทันที
    matched_province = None
    for kw, prov in PROVINCE_MAP.items():
        if kw in query:
            matched_province = prov
            break
            
    if matched_province:
        for meta in doc_meta:
            if matched_province in meta["source"] or matched_province in meta["snippet"]:
                results.append({
                    "score": 1.0,
                    "source": meta["source"],
                    "content": meta["snippet"]
                })
        if results:
            return results

    # 2. ค้นหาด้วย Vector Search ถ้าไม่เจอชื่อจังหวัดโดยตรง
    model = load_embedding_model()
    q_vec = model.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, top_k)
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1:
            snippet = doc_meta[idx]["snippet"]
            results.append({
                "score": float(score),
                "source": doc_meta[idx]["source"],
                "content": snippet
            })
    return results

# ----------------- Chat Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "สบายดีครับพี่น้อง! ยินดีต้อนรับสู่แดนอีสานบ้านเฮาเด้อครับ 🌾 ข้อยคือ AI ไกด์นำเที่ยว 20 จังหวัดภาคอีสาน ถามได้เลยทั้งพิกัดเที่ยว เวลาเปิด-ปิด ร้านแซ่บ หรือของฝากประจำถิ่น พิมพ์ถามหรือคลิกปุ่มแนะนำด้านบนได้เลยครับ!"
        }
    ]

# ----------------- ปุ่มคำถามแนะนำยอดนิยม 12 ข้อ -----------------
st.markdown('<div class="quick-title">💡 คำถามแนะนำยอดนิยม (คลิกเพื่อถามทันที):</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
sample_questions = [
    "แนะนำที่เที่ยวอุบลราชธานี พร้อมพิกัดและเวลาเปิด-ปิด",
    "ร้านอาหารเด็ดอุบลราชธานี มีร้านไหนแซ่บๆ บ้าง",
    "ไปเที่ยวเลย มีกิจกรรมอะไรที่ต้องทำหรือห้ามพลาด",
    "อนุสาวรีย์ท้าวสุรนารี (ย่าโม) โคราช ตั้งอยู่ที่ไหนและเปิดเวลาใด",
    "ร้านไข่กระทะขึ้นชื่อในอุดรธานีเปิดกี่โมง",
    "ไปบุรีรัมย์ ซื้อของฝากอะไรดี",
    "เวลาเปิด-ปิด ผาแต้ม อุบลราชธานี",
    "วัดภูทอก บึงกาฬ เปิดกี่โมงและปิดช่วงไหน",
    "ของฝากขึ้นชื่อของยโสธรมีอะไรบ้าง",
    "สกายวอล์ค วัดผาตากเสื้อ หนองคาย อยู่ที่ไหน",
    "มีทัวร์ดำน้ำดูปะการังที่กาฬสินธุ์ไหม",
    "กระเช้าลอยฟ้าขึ้นยอดภูเขาไฟร้อยเอ็ดราคาเท่าไหร่"
]

selected_query = None
for idx, q in enumerate(sample_questions):
    target_col = col1 if idx % 2 == 0 else col2
    if target_col.button(f"📌 {q}", key=f"btn_{idx}"):
        selected_query = q

# ----------------- Render ประวัติการแชท -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- รับ Input จาก Chat หรือ ปุ่มกด -----------------
user_input = st.chat_input("พิมพ์คำถาม เช่น เที่ยวอุบล, ของกินขอนแก่น...")
active_query = selected_query if selected_query else user_input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    retrieved_docs = search_context(active_query)

    context_blocks = [
        f"[{doc['source']}]:\n{doc['content']}"
        for doc in retrieved_docs
    ]
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "คุณคือ 'AI ไกด์นำเที่ยวอีสานบ้านเฮา' ผู้เชี่ยวชาญการท่องเที่ยว 20 จังหวัดภาคอีสานของประเทศไทย\n"
        "บุคลิกภาพ: เป็นมิตร อารมณ์ดี สุภาพ ตอบกระชับ เนื้อหาเน้นๆ มีคำสร้อยอีสานน่ารักๆ เช่น 'เด้อครับ', 'น้อครับ', 'แซ่บหลาย', 'อีหลี'\n\n"
        "กฎเหล็กการตอบคำถาม:\n"
        "1. **ห้ามถามย้อนกั๊กคำตอบโดยเด็ดขาด**: เมื่อผู้ใช้ถามถึงจังหวัดใด (เช่น 'เที่ยวอุบล', 'แนะนำที่เที่ยวขอนแก่น') ให้ **ตอบข้อมูลเนื้อหาจริงทันที** โดยสรุปสถานที่ท่องเที่ยวไฮไลท์ 3-5 แห่ง พร้อมพิกัดและเวลาเปิด-ปิดให้ชัดเจน\n"
        "2. ตอบแยกหมวดหมู่อย่างเป็นระเบียบ เช่น 📍 สถานที่ท่องเที่ยวแนะนำ, 🍲 ร้านเด็ดต้องแวะ, 🎁 ของฝากขึ้นชื่อ\n"
        "3. ตบท้ายคำตอบสั้นๆ 1 ประโยคอย่างเป็นกันเอง เช่น 'หากอยากให้ข้อยเจาะลึกสายกินแซ่บๆ หรือสายบุญวัดงามๆ เพิ่มเติม บอกข้อยได้เลยเด้อครับ!'\n"
        "4. หากถามถึงสิ่งที่ไม่มีจริงในภาคอีสาน (เช่น ดำน้ำดูปะการัง, นั่งกระเช้าลอยฟ้า): ให้ตอบปฏิเสธสุภาพว่า 'ขออภัยเด้อครับ ข้อยบ่พบข้อมูลนี้ในเอกสารท่องเที่ยว' และอธิบายสั้นๆ ว่าไม่มีกิจกรรมนี้ในจังหวัดดังกล่าวครับ"
    )

    user_query_payload = f"ข้อมูลอ้างอิง (Context):\n{context_str}\n\nคำถามของผู้ใช้: {active_query}"

    with st.chat_message("assistant"):
        with st.spinner("ไกด์กำลังจัดทริปให้เด้อครับ..."):
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
                    max_tokens=800
                )
                bot_reply = response.choices[0].message.content
                st.markdown(bot_reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_reply
                })

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Groq API: {str(e)}")

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