# 🌾 ISAN AI Travel Guide (คู่มือท่องเที่ยว 20 จังหวัดภาคอีสานด้วย RAG)

ระบบผู้ช่วย AI แนะนำการท่องเที่ยว 20 จังหวัดภาคอีสาน พัฒนาด้วยเทคนิค **Retrieval-Augmented Generation (RAG)** ผสานการค้นหาแบบ **Hybrid Search** ร่วมกับ **Groq API** และถ่ายทอดคำตอบผ่านคาแรคเตอร์ **"ไกด์อีสานบ้านเฮา"** ที่เป็นกันเอง แม่นยำตามเอกสารอ้างอิง และมีหน้าเว็บที่ทันสมัยใช้งานง่าย

---

## 📌 จุดเด่นของระบบ (Features)

* **ฐานข้อมูล 20 จังหวัดอีสานครบถ้วน**:
  * 📍 **12 สถานที่ท่องเที่ยวต่อจังหวัด** (ระบุพิกัดที่ตั้ง และเวลาเปิด-ปิด)
  * 🍲 **5 ร้านอาหารขึ้นชื่อต่อจังหวัด** (พร้อมพิกัดและเวลาทำการ)
  * 🎁 **ของฝากและของดีประจำจังหวัด** (OTOP / สินค้า GI)
  * 🎯 **ไฮไลท์กิจกรรมห้ามพลาด (Must-Do Activities)** เมื่อไปเยือนจังหวัดนั้นๆ
* **ระบบสืบค้นแบบ Hybrid Search**: ผสาน Keyword Matching ของ 20 จังหวัด (รองรับชื่อย่อ เช่น โคราช, อุบล, สารคาม) ร่วมกับ FAISS Vector Search (`paraphrase-multilingual-MiniLM-L12-v2`) ทำให้ค้นหาข้อมูลได้แม่นยำ ไม่ตกหล่น
* **ระบบ Auto-Detect Groq Model**: ตรวจจับโมเดลที่ใช้งานได้จริงในบัญชีโดยอัตโนมัติ ป้องกันปัญหา Model Decommissioned
* **คาแรคเตอร์ "ไกด์อีสานบ้านเฮา"**: เว้าสำเนียงอีสานน่ารัก เป็นกันเอง ตอบข้อมูลจริงทันทีไม่กั๊กคำตอบ พร้อมตอบคำถามแยกตามไลฟ์สไตล์ (สายกิน, สายเที่ยวธรรมชาติ, สายบุญวัฒนธรรม)
* **ตอบปฏิเสธข้อมูลที่ไม่มีจริง (Hallucination Prevention)**: ปฏิเสธอย่างถูกต้องเมื่อถามกิจกรรมที่ไม่มีในเอกสาร เช่น ดำน้ำดูปะการัง หรือกระเช้าลอยฟ้า
* **Interactive UI**: ออกแบบด้วย Streamlit สไตล์โมเดิร์น พร้อมปุ่มกดคำถามแนะนำยอดนิยม 12 ข้อ และระบบ Auto-scroll เลื่อนหาคำตอบอัตโนมัติ

---

## 🛠️ สถาปัตยกรรมและเทคโนโลยีที่ใช้ (Tech Stack)

* **Language**: Python 3.10+
* **Frontend / Framework**: Streamlit
* **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Vector Database**: FAISS (Facebook AI Similarity Search)
* **LLM Engine**: Groq API (Llama 3.3 / Llama 3.1 / Qwen / Mixtral)

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
isan-rag-guide/
├── .streamlit/
│   └── secrets.toml          # ไฟล์เก็บ GROQ_API_KEY (ไม่นำขึ้น Git)
├── data/                     # คลังข้อมูล 20 จังหวัดภาคอีสาน (.txt)
│   ├── 01_nakhon_ratchasima.txt
│   ├── 02_khon_kaen.txt
│   └── ... (ครบ 20 จังหวัด)
├── .gitignore
├── app.py                    # โค้ดหลักของ Streamlit App และ RAG Pipeline
├── generate_data.py          # สคริปต์สำหรับสร้างชุดข้อมูล 20 จังหวัด
├── requirements.txt          # รายการไลบรารีที่จำเป็น
└── README.md                 # เอกสารอธิบายโปรเจกต์