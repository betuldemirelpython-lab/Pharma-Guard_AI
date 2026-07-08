# 🛡️ Pharma-Guard AI

**Yapay Zeka Destekli Akıllı İlaç Denetim ve Doğrulama Asistanı**

Pharma-Guard AI, ilaç kutularının fotoğraflarını veya isimlerini kullanarak saniyeler içinde kapsamlı denetim yapan, çoklu yapay zeka ajanlarından (Multi-Agent) oluşan bir güvenlik asistanıdır. Sistem, Gemini ve Groq (Llama 3) modellerini bir arada kullanarak hem görsel okuma hem de tıbbi analiz süreçlerini yönetir.

---

## ✨ Özellikler

- **📸 Görsel Analiz (Vision Scanner):** İlaç kutusunun fotoğrafından ilacın adını, mg değerini ve etken maddesini yüksek doğrulukla okur (Gemini 2.0 Flash Vision).
- **🔎 Hibrit RAG ve İnternet Taraması:** Kendi PDF veritabanınızdaki prospektüsleri tarar. Eğer veritabanında belge yoksa, gerçek zamanlı olarak DuckDuckGo üzerinden internetteki güncel ve güvenilir prospektüs verilerini analiz eder.
- **🛡️ Güvenlik Denetimi:** İlacın dozaj, kullanım amacı ve yan etkilerinde bir uyuşmazlık olup olmadığını çapraz denetler. Yanlış kutu ve hatalı dozajları tespit ederek "VERİ UYUŞMAZLIĞI ALARMI" verir.
- **📚 Kendi Kendine Öğrenme:** İnternet üzerinden denetlenen ilaçların raporları tek tuşla sisteme öğretilir (Vektör veritabanına RAG PDF'si olarak eklenir).
- **📥 Rapor Çıktısı:** Saniyeler içinde Profesyonel PDF Raporu oluşturur ve indirmenizi sağlar.

---

## 🏗️ Sistem Mimarisi (Ajanlar)

1. **VisionScannerAgent:** Görseldeki metinleri (OCR) ve ilaç kutusunu analiz eder.
2. **RAGSpecialistAgent:** Kendi lokal vektör veritabanınızdan veya canlı internet aramalarından ilaca ait prospektüs bilgilerini çeker.
3. **SafetyAuditorAgent:** Elde edilen veriler ile görseldeki verileri karşılaştırarak potansiyel riskleri ve uyuşmazlıkları denetler.
4. **CorporateAnalystAgent:** İlacın üretici firmasının güvenilirliğini kontrol eder.
5. **ReportSynthesizerAgent:** Tüm ajanların verilerini birleştirip nihai, düzenli bir markdown raporu oluşturur.

---

## 🚀 Kurulum

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
```bash
git clone https://github.com/betuldemirelpython-lab/Pharma-Guard_AI.git
cd Pharma-Guard_AI
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. (Opsiyonel) Çevresel değişkenler için ana dizinde bir `.env` dosyası oluşturun ve API anahtarlarınızı ekleyin:
```env
GEMINI_API_KEY=sizin_gemini_anahtariniz
GROQ_API_KEY=sizin_groq_anahtariniz
```
*Not: Streamlit arayüzünde yan panelden de bu anahtarları anlık olarak girebilirsiniz.*

4. Uygulamayı Başlatın:
```bash
streamlit run app.py
```

---

## 💡 Kullanım

- Sol panelden bir **İlaç Kutusu Fotoğrafı** yükleyin.
- Fotoğrafı yükledikten sonra "🚀 Denetimi Başlat" butonuna tıklayın.
- Ya da sadece ilacın ismini denetlemek istiyorsanız "Sadece İlaç Adı ile Sorgula" panelini açıp isim yazabilirsiniz.
- Rapor oluştuktan sonra "📥 Profesyonel PDF Raporu İndir" ile indirebilir veya "💾 Raporu RAG Kaynakçaya Ekle" butonuna basarak yapay zekanın o ilacı öğrenmesini sağlayabilirsiniz.

---

## 👩‍💻 Proje Geliştiricisi
**Betül Altınkaynak Demirel**
