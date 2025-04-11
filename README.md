# CV Bias Analyzer Tool

Bu araç, özgeçmiş tarama ve işe alım süreçlerindeki potansiyel biasları (yanlılıkları) tespit etmek, analiz etmek ve azaltmak için geliştirilmiş kapsamlı bir uygulamadır. EU AI Act kapsamında, işe alım sistemleri yüksek riskli AI uygulamaları arasında değerlendirilmektedir.

## Özellikler

- **PDF, DOCX ve CSV dosya desteği**: Farklı formatlardaki CV'leri analiz edebilme
- **IBM AI Fairness 360 entegrasyonu**: Bias metrikleri hesaplama ve azaltma algoritmaları
- **Microsoft Fairlearn entegrasyonu**: Fairness ölçümleri ve model performans analizleri
- **SHAP & LIME açıklanabilirlik**: Model kararlarının şeffaflığını sağlama
- **Kapsamlı görselleştirmeler**: Bias analizini anlaşılır şekilde gösterme
- **EU AI Act rehberi**: Yüksek riskli AI sistemleri için uyumluluk gereksinimleri

## Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/ysntns/cv-bias-analyzer.git
cd cv-bias-analyzer

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Uygulamayı çalıştırın
streamlit run app.py
```

## Kullanım

Uygulama, altı ana sekme sunmaktadır:

1. **Veri Analizi**: Yüklenen veya örnek veri setinin temel analizleri
2. **Bias Analizi**: Disparate Impact ve diğer fairness metriklerinin hesaplanması
3. **Model Analizi**: Farklı demografik gruplar için model performans değerlendirmesi
4. **Açıklanabilirlik**: SHAP, LIME ve ELI5 ile model kararlarının açıklanması 
5. **EU AI Act Rehberi**: Yasal gereksinimler ve uyumluluk adımları
6. **Deployment**: Farklı ortamlarda nasıl dağıtılabileceğine dair rehber

## Lisans

MIT
