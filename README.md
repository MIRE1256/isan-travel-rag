# 🧭 คู่มือท่องเที่ยว 20 จังหวัดภาคอีสาน (Isan Travel RAG GuideBot)

เว็บแอปพลิเคชันถาม-ตอบข้อมูลสถานที่ท่องเที่ยว เวลาเปิด-ปิด พิกัดที่ตั้ง ร้านอาหาร และของฝากครบ 20 จังหวัดภาคอีสาน ด้วยเทคนิค Retrieval-Augmented Generation (RAG)

## 📌 วัตถุประสงค์
เป็นผู้ช่วยอัจฉริยะสำหรับนักท่องเที่ยว ให้ข้อมูลสถานที่ท่องเที่ยว 8 แห่ง ร้านอาหาร 2 แห่ง และของฝากในทุกจังหวัดของภาคอีสาน โดย AI จะตอบเฉพาะข้อมูลจากเอกสารคลังความรู้เท่านั้น เพื่อตัดปัญหา Hallucination

## 🛠️ Data Sources
- ข้อมูลท่องเที่ยว การท่องเที่ยวแห่งประเทศไทย (ททท. สำนักงานภาคอีสาน)
- ข้อมูลอุทยานแห่งชาติ จากกรมอุทยานแห่งชาติ สัตว์ป่า และพันธุ์พืช
- ฐานข้อมูลวัฒนธรรมและโบราณสถาน จากกรมศิลปากร

## ⚙️ RAG Architecture
- **Chunking:** แบ่งท่อนข้อความขนาด 400 ตัวอักษร ซ้อนทับ 80 ตัวอักษร
- **Embedding:** โมเดล `paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Search:** `faiss-cpu` (Inner Product)
- **LLM:** Groq API (`llama3-8b-8192`) ตั้งค่า `temperature=0.1`

## 🚀 วิธีการทดสอบในเครื่อง (Local Setup)
```bash
git clone <URL_REPO>
cd isan-rag-guide
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
mkdir .streamlit
echo 'GROQ_API_KEY = "your-api-key"' > .streamlit/secrets.toml
streamlit run app.py