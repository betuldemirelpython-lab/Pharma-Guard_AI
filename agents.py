import os
import json
import traceback
from PIL import Image
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

from utils import query_rag

load_dotenv()

# Configure APIs
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

print(f"[Pharma-Guard-AI] GEMINI_API_KEY yüklendi mi: {'EVET' if gemini_key else 'HAYIR'}")
print(f"[Pharma-Guard-AI] GROQ_API_KEY yüklendi mi: {'EVET' if groq_key else 'HAYIR'}")

if gemini_key:
    genai.configure(api_key=gemini_key)

groq_client = None
if groq_key:
    groq_client = Groq(api_key=groq_key)

def get_gemini_model():
    # Using gemini-2.0-flash as it's the recommended multimodal orchestrator
    return genai.GenerativeModel("gemini-2.0-flash")

def extract_json(text):
    try:
        # Find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        return json.loads(text)
    except Exception as e:
        print(f"[Pharma-Guard-AI] JSON extract/parse error: {e}. Original text: {text}")
        raise e

class VisionScannerAgent:
    """
    Scans the image of the medicine package using Gemini 2.0 Flash to extract name, active ingredient, dosage, and manufacturer.
    """
    def run(self, image: Image.Image) -> dict:
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Optimize image size for faster API response
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"[Pharma-Guard-AI] Görsel boyutu optimize edildi: {image.size}")

        prompt = """
        You are a highly precise medical vision assistant. Analyze the uploaded image of the medicine package.
        
        Extract the following information and output it EXACTLY in JSON format:
        {
          "ilac_adi": "İlaç markası/ticari adı (e.g. Parol)",
          "etken_madde": "Etken madde ismi (e.g. Parasetamol)",
          "dozaj": "Dozaj değeri ve birimi (e.g. 500 mg)",
          "uretici_firma": "Üretici firma adı",
          "yazi_okunuyor_mu": true,
          "guven_puani": 9
        }
        
        CRITICAL RULES:
        1. If the text on the package is blurry, unreadable, or missing, set "yazi_okunuyor_mu" to false. Do NOT guess the name of the medicine.
        2. Keep the output as valid, clean JSON only. Do not add introductory or concluding text.
        """
        try:
            model = get_gemini_model()
            response = model.generate_content([image, prompt])
            print(f"[Pharma-Guard-AI] Gemini Vision Yanıtı:\n{response.text}")
            
            data = extract_json(response.text.strip())
            return data
        except Exception as e:
            print(f"[Pharma-Guard-AI] VisionScannerAgent error: {e}")
            traceback.print_exc()
            return {
                "ilac_adi": "Bilinmeyen İlaç",
                "etken_madde": "Bilinmeyen",
                "dozaj": "0 mg",
                "uretici_firma": "Bilinmeyen",
                "yazi_okunuyor_mu": False,
                "guven_puani": 1,
                "hata": str(e)
            }

class RAGSpecialistAgent:
    """
    Retrieves prospectus data using RAG from local PDFs.
    """
    def run(self, ilac_adi: str, etken_madde: str) -> dict:
        query = f"{ilac_adi} {etken_madde} prospektüs endikasyonlar yan etkiler dozaj uyarısı"
        try:
            rag_docs = query_rag(query, k=5)
            return {
                "bulunan_belgeler": rag_docs,
                "durum": "BAŞARILI" if rag_docs else "KAYNAK_BULUNAMADI"
            }
        except Exception as e:
            return {
                "bulunan_belgeler": [],
                "durum": f"HATA: {str(e)}"
            }

class SafetyAuditorAgent:
    """
    Validates scanned medicine parameters against safety and RAG rules.
    Checks for inconsistencies (e.g., dosage, active ingredients) and identifies side effects.
    """
    def run(self, vision_data: dict, rag_data: dict) -> dict:
        # Check if RAG details match detected details
        matching_rag_docs = rag_data.get("bulunan_belgeler", [])
        
        if not matching_rag_docs:
            return {
                "uyumlu_mu": False,
                "durum": "VERİ UYUŞMAZLIĞI (Prospektüs Bulunamadı)",
                "guven_puani": 2,
                "uyarilar": ["Sistem veritabanında bu ilaca ait onaylı bir prospektüs bulunamadı!"]
            }

        # Combine text for analysis
        combined_prospectus = "\n".join([doc["content"] for doc in matching_rag_docs])
        
        # Let's run safety analysis with Groq/Gemini
        prompt = f"""
        Görev: İlaç Denetimi ve Güvenlik Analizi
        
        [Tarama Verileri (Vision)]:
        - İlaç Adı: {vision_data.get('ilac_adi')}
        - Etken Madde: {vision_data.get('etken_madde')}
        - Dozaj: {vision_data.get('dozaj')}
        
        [RAG Prospektüs Verileri]:
        {combined_prospectus[:4000]}
        
        Lütfen aşağıdaki kurallara göre analizi yapın:
        1. Tarama verilerindeki etken madde ve dozaj değeri, prospektüs verileri ile uyumlu mu?
        2. Prospektüsteki dozaj/mg bilgisiyle en ufak bir fark var mı? (Örn: 1 mg fark olsa bile uyuşmazlık sayılmalıdır).
        3. İlacın "Yan Etkileri", "Diğer İlaçlarla Etkileşimi" ve "Kimler Kullanamaz" bilgilerini özetleyin.
        
        Yanıtı şu JSON formatında dönün:
        {{
          "uyumlu_mu": true/false,
          "durum": "VERİ UYUMLU" veya "VERİ UYUŞMAZLIĞI",
          "fark_detaylari": "Uyuşmazlık veya fark varsa açıklaması",
          "yan_etkiler": ["Yan etki 1", "Yan etki 2"],
          "etkilesimler": ["Etkileşim 1", "Etkileşim 2"],
          "kontrendikasyonlar": ["Kimler kullanamaz 1", "Kimler kullanamaz 2"],
          "guven_puani": 1-10 aralığında puan
        }}
        
        Sadece geçerli JSON çıktısı verin.
        """
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            return extract_json(response.text.strip())
        except Exception as e:
            return {
                "uyumlu_mu": False,
                "durum": f"ANALİZ HATASI: {str(e)}",
                "guven_puani": 1,
                "yan_etkiler": [],
                "etkilesimler": [],
                "kontrendikasyonlar": []
            }

class CorporateAnalystAgent:
    """
    Analyzes manufacturer history and certificates.
    """
    def run(self, uretici_firma: str) -> dict:
        prompt = f"""
        Aşağıdaki ilaç üretici firmasını analiz et ve geçmişi, sertifikaları hakkında kısa bilgi ver.
        Firma: {uretici_firma}
        
        Çıktı Formatı JSON:
        {{
          "firma_tanimi": "Firma hakkında özet bilgi",
          "bilinen_sertifikalar": ["GMP", "ISO 9001 vs."],
          "guvenirlik_puani": 1-10 aralığında puan
        }}
        """
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            return extract_json(response.text.strip())
        except Exception as e:
            return {
                "firma_tanimi": f"{uretici_firma} hakkında ek bilgi alınamadı.",
                "bilinen_sertifikalar": ["Bilinmiyor"],
                "guvenirlik_puani": 5
            }

class ReportSynthesizerAgent:
    """
    Combines outputs from all agents and produces the final audit report.
    Uses Llama-3-70B via Groq (with fallback to Gemini if Groq is unavailable).
    """
    def run(self, vision: dict, rag: dict, safety: dict, corporate: dict) -> str:
        # Format input for LLM
        prompt = f"""
        ### ROLE: PHARMA-GUARD MASTER ORCHESTRATOR (PG-MO) ###
        Görevin vatandaşın ilacı yanlış kullanmasını önlemektir. Tüm alt ajanlardan gelen verileri birleştirerek nihai denetim raporunu oluştur.
        
        [Vision-Scanner Verileri]:
        {json.dumps(vision, ensure_ascii=False, indent=2)}
        
        [RAG-Specialist Verileri (Bulunan Prospektüsler)]:
        {json.dumps([{ 'source': d['source'], 'page': d['page'] } for d in rag.get('bulunan_belgeler', [])], ensure_ascii=False, indent=2)}
        
        [Safety-Auditor Verileri]:
        {json.dumps(safety, ensure_ascii=False, indent=2)}
        
        [Corporate-Analyst Verileri]:
        {json.dumps(corporate, ensure_ascii=False, indent=2)}
        
        ### OPERASYONEL PROTOKOLLER VE KISITLAMALAR:
        - GÜVEN PUANI (Confidence Score): Her bilgi parçası için 1-10 arası bir puan ver.
        - HALÜSİNASYON ENGELİ: Eğer ilacın etken maddesi ile prospektüs bilgisi eşleşmiyorsa veya uyuşmazlık varsa raporun en üstünde kırmızı renkle 'VERİ UYUŞMAZLIĞI ALARMI' başlığını kullan ve detaylı uyar.
        - DİL VE ÜSLUP: Rapor tamamen Türkçe, tıbbi terimleri parantez içinde açıklayan, net bir dil içermeli.
        - İNDİRME FORMATI: Çıktıyı "İndirilebilir Rapor" için uygun bir Markdown hiyerarşisinde oluştur.
        
        ### ÇIKTI HİYERARŞİSİ (ZORUNLU):
        1. İlaç Kimlik Özeti
        2. Kullanım Amacı (Endikasyonlar)
        3. Kritik Uyarılar ve Yan Etkiler
        4. Etken Madde ve Üretici Detayları
        5. RAG / Kaynakça (Hangi belgeden, hangi sayfadan alındı?)
        """
        
        # Try Groq (Llama-3-70b)
        if groq_client:
            try:
                response = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Sen profesyonel bir tıp ve ilaç denetleme asistanısın. Türkçe yanıt ver."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-70b-8192",
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Groq API error, falling back to Gemini: {e}")
        
        # Fallback to Gemini
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Rapor Sentezi Oluşturulurken Hata Oluştu: {e}"

class PharmaOrchestrator:
    """
    Main manager coordinating all agents in sequence.
    """
    def __init__(self):
        self.vision_scanner = VisionScannerAgent()
        self.rag_specialist = RAGSpecialistAgent()
        self.safety_auditor = SafetyAuditorAgent()
        self.corporate_analyst = CorporateAnalystAgent()
        self.report_synthesizer = ReportSynthesizerAgent()

    def process_medicine(self, image: Image.Image) -> tuple[str, dict]:
        # Step 1: Scan image
        vision_result = self.vision_scanner.run(image)
        
        if not vision_result.get("yazi_okunuyor_mu", True):
            return "KURAL İHLALİ: Yazı okunmuyor veya ilaç kutusu tespit edilemedi. Lütfen fotoğrafı daha ışıklı bir yerde çekerek tekrar yükleyin.", {"vision": vision_result}

        # Step 2: RAG Search
        rag_result = self.rag_specialist.run(
            vision_result.get("ilac_adi"), 
            vision_result.get("etken_madde")
        )

        # Step 3: Safety Audit
        safety_result = self.safety_auditor.run(vision_result, rag_result)

        # Step 4: Corporate analysis
        corporate_result = self.corporate_analyst.run(vision_result.get("uretici_firma"))

        # Step 5: Synthesize Report
        final_report = self.report_synthesizer.run(
            vision_result, 
            rag_result, 
            safety_result, 
            corporate_result
        )

        agent_logs = {
            "vision": vision_result,
            "rag": {
                "durum": rag_result.get("durum"),
                "bulunan_belgeler_sayisi": len(rag_result.get("bulunan_belgeler", []))
            },
            "safety": safety_result,
            "corporate": corporate_result
        }

        return final_report, agent_logs
