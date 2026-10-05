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
    page_title="ไกด์ท่องเที่ยว 20 จังหวัดภาคอีสาน - ISAN AI Guide",
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
    <div class="hero-subtitle">พาเลาะ 20 จังหวัดอีสานบ้านเฮา 🌾 ครบทั้งพิกัดเที่ยว ร้านแซ่บ ของฝาก และไฮไลท์ห้ามพลาด คัดสรร 12 ที่เที่ยว 5 ร้านเด็ด ของฝาก และกิจกรรมห้ามพลาด พร้อมทริปสายกิน สายชิล สายวัฒนธรรม จัดเต็มทุกสาย!</div>
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
    "นครราชสีมา", "โคราช", "ขอนแก่น", "อุดรธานี", "อุบลราชธานี", "อุบล", "นครพนม", "เลย",
    "หนองคาย", "ร้อยเอ็ด", "บุรีรัมย์", "บึงกาฬ", "สกลนคร", "สุรินทร์", "ศรีสะเกษ",
    "กาฬสินธุ์", "ชัยภูมิ", "มหาสารคาม", "มุกดาหาร", "ยโสธร", "หนองบัวลำภู", "อำนาจเจริญ"
]

def search_context(query: str, top_k: int = 6):
    results = []
    
    # ดึงข้อมูลตรงตามจังหวัด และแนวคำค้นหา (สายกิน, สายเที่ยว, สายวัฒนธรรม)
    for meta in doc_meta:
        chunk_text = meta["snippet"]
        for prov in ISAN_PROVINCES:
            if prov in query:
                # แปลงคำค้น 'อุบล' เป็น 'อุบลราชธานี'
                match_prov = "อุบลราชธานี" if prov == "อุบล" else ("นครราชสีมา" if prov == "โคราช" else prov)
                if match_prov in chunk_text or match_prov in meta["source"]:
                    # หากเป็นสายกิน
                    if any(kw in query for kw in ["สายกิน", "กิน", "ร้านอาหาร", "ของกิน", "แซ่บ"]) and "ร้านอาหาร" in chunk_text:
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})
                    # หากเป็นของฝาก
                    elif any(kw in query for kw in ["ของฝาก", "ของดี", "ซื้อ"]) and "ของฝาก" in chunk_text:
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})
                    # หากเป็นสายเที่ยว / วัฒนธรรม
                    elif any(kw in query for kw in ["สายเที่ยว", "สายวัฒนธรรม", "สายบุญ", "วัด", "ที่เที่ยว", "สถานที่", "ต้องทำ"]) and any(h in chunk_text for h in ["สถานที่ท่องเที่ยว", "สิ่งที่เมื่อมาจังหวัดนี้"]):
                        results.append({"score": 1.0, "source": meta["source"], "content": chunk_text})
                    else:
                        results.append({"score": 0.9, "source": meta["source"], "content": chunk_text})

    # Vector Semantic Search ช่วยเก็บตก
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
    return results[:7]

# ----------------- Chat Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "สบายดีครับพี่น้อง! ยินดีต้อนรับสู่แดนอีสานบ้านเฮาเด้อครับ 🌾 ข้อยคือ AI ไกด์นำเที่ยว 20 จังหวัดภาคอีสาน อยากไปเลาะไส ถามพิกัด ที่เที่ยว เวลาเปิด-ปิด ร้านอาหารแซ่บๆ หรือบอกสไตล์มาได้เลย เช่น **'สายกินอุบล'**, **'สายวัฒนธรรมสกลนคร'**, หรือ **'สายลุยเลย'** กะจัดให้ได้หมดเด้อครับ!"
        }
    ]

# ----------------- ปุ่มคำถามแนะนำยอดนิยม 12 ข้อ -----------------
st.markdown('<div class="quick-title">💡 คำถามแนะนำยอดนิยม (คลิกเพื่อถามทันที):</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
sample_questions = [
    "สายกินอุบลราชธานี มีร้านเด็ดร้านไหนต้องแวะบ้าง",
    "สายวัฒนธรรมขอนแก่น แนะนำที่เที่ยวและวัดสวยๆ หน่อย",
    "สายเที่ยวธรรมชาติจังหวัดเลย มีที่ไหนห้ามพลาด",
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
user_input = st.chat_input("พิมพ์คำถาม เช่น สายกินอุบล, ที่เที่ยวโคราช...")
active_query = selected_query if selected_query else user_input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    retrieved_docs = search_context(active_query, top_k=6)

    context_blocks = [
        f"[{doc['source']}]:\n{doc['content']}"
        for doc in retrieved_docs
    ]
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "คุณคือ 'AI ไกด์นำเที่ยวอีสานบ้านเฮา' ผู้เชี่ยวชาญการท่องเที่ยว 20 จังหวัดภาคอีสานของประเทศไทย\n"
        "บุคลิกและสไตล์การพูด:\n"
        "- ร่าเริง อารมณ์ดี สุภาพ เป็นกันเอง เว้าภาษาไทยมาตรฐานปนสำเนียงอีสานน่ารักๆ เช่น 'เด้อครับ', 'น้อครับ', 'แซ่บหลาย', 'อีหลี', 'ไปเลาะ'\n\n"
        "กฎการตอบแบบแยกสายท่องเที่ยว (สำคัญมาก):\n"
        "1. หากผู้ใช้ถามถึง 'สายกิน' (เช่น สายกินอุบล): ให้คัดเลือกร้านอาหารขึ้นชื่อ 5 แห่งและของกินเด็ดในบริบทมาแนะนำ บอกพิกัดและเวลาเปิด-ปิดชัดเจน\n"
        "2. หากผู้ใช้ถามถึง 'สายเที่ยวธรรมชาติ' หรือ 'สายลุย': ให้เลือกแหล่งท่องเที่ยวธรรมชาติ ภูเขา น้ำตก จุดชมวิว พร้อมเวลาทำการและไฮไลท์ที่ต้องทำ\n"
        "3. หากผู้ใช้ถามถึง 'สายวัฒนธรรม' หรือ 'สายบุญ': ให้เลือกวัดสำคัญ โบราณสถาน และวิถีชุมชนผ้าทอมาแนะนำอย่างสวยงาม\n"
        "4. หากผู้ใช้พิมพ์มาแบบกว้างๆ (เช่น 'ไปอุบล', 'เที่ยวขอนแก่น', 'แนะนำหน่อย'): ให้สรุปภาพรวมสั้นๆ แล้ว **ถามผู้ใช้กลับอย่างเป็นกันเองเสมอ** ว่า: 'อยากให้ข้อยจัดทริปแนวไหนดีครับ? มีทั้ง 🍜 สายกินแซ่บๆ, 🌿 สายเที่ยวธรรมชาติ, หรือ 🛕 สายบุญวัฒนธรรม ลองบอกข้อยได้เลยเด้อครับ!'\n"
        "5. สำหรับสถานที่/กิจกรรมที่ไม่มีจริงในภาคอีสาน (เช่น ดำน้ำดูปะการัง, กระเช้าลอยฟ้าภูเขาไฟ): ให้ตอบปฏิเสธสุภาพปนน่ารักว่า 'ขออภัยเด้อครับ ข้อยบ่พบข้อมูลนี้ในเอกสารท่องเที่ยว' และชี้แจงว่าไม่มีกิจกรรมนี้ในจังหวัดดังกล่าวครับ"
    )

    user_query_payload = f"ข้อมูลอ้างอิง (Context):\n{context_str}\n\nคำถาม: {active_query}"

    with st.chat_message("assistant"):
        with st.spinner("ไกด์กำลังคัดร้านแซ่บกับพิกัดเด็ดให้เด้อครับ..."):
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
                    max_tokens=700
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