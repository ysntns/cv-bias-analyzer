import streamlit as st

# Bu komut en başta olmalı - diğer streamlit komutlarından önce
st.set_page_config(page_title="Kapsamlı CV Bias Analiz Aracı", layout="wide", 
                  page_icon="📊", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import tempfile
import io
import re
import warnings
warnings.filterwarnings('ignore')

# Bias analiz kütüphaneleri
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    from aif360.algorithms.postprocessing import CalibratedEqualizedOdds
    aif360_available = True
except ImportError:
    aif360_available = False
    st.warning("AIF360 kütüphanesi yüklenmemiş. Tam bias analizi için: pip install aif360")

# Fairlearn entegrasyonu
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import ExponentiatedGradient, GridSearch
    from fairlearn.widget import FairlearnDashboard
    fairlearn_available = True
except ImportError:
    fairlearn_available = False
    st.warning("Fairlearn kütüphanesi yüklenmemiş. Fairness metrikleri için: pip install fairlearn")

# Açıklanabilirlik kütüphaneleri
try:
    import shap
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    import eli5
    from eli5.sklearn import PermutationImportance
    explainability_available = True
except ImportError:
    explainability_available = False
    st.warning("SHAP, LIME veya ELI5 kütüphanesi yüklenmemiş. Model açıklanabilirliği için bunları yükleyin.")

# PDF ve DOCX işleme kütüphaneleri
try:
    import PyPDF2
    import docx
    pdf_docx_support = True
except ImportError:
    pdf_docx_support = False
    st.warning("PDF/DOCX desteği eksik. PDF ve DOCX dosyalarını işlemek için: pip install PyPDF2 python-docx")


# Uygulama başlığı ve açıklama
st.title("📊 Kapsamlı CV Bias Analiz Aracı")
st.write("""
Bu gelişmiş araç, özgeçmiş tarama ve işe alım süreçlerindeki bias (yanlılık) ve fairness (adillik) sorunlarını tespit etmenize, 
analiz etmenize ve azaltmanıza yardımcı olur. EU AI Act kapsamında işe alım sistemleri yüksek riskli AI uygulamaları 
arasında değerlendirilmektedir.

**Desteklenen Formatlar:** CSV, PDF, DOCX ve TXT
""")

# ---- Yardımcı Fonksiyonlar ----
def extract_text_from_pdf(file_obj):
    """PDF dosyasından metin çıkarır"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"PDF işleme hatası: {str(e)}")
        return ""

def extract_text_from_docx(file_obj):
    """DOCX dosyasından metin çıkarır"""
    try:
        doc = docx.Document(file_obj)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)
    except Exception as e:
        st.error(f"DOCX işleme hatası: {str(e)}")
        return ""

def extract_demographics(text):
    """Metinden demografik özellikleri çıkarır"""
    demographics = {}
    
    # Cinsiyet tespiti
    male_keywords = ['bay', 'erkek', 'adam', 'bey', 'mr', 'male', 'he', 'his']
    female_keywords = ['bayan', 'kadın', 'hanım', 'ms', 'mrs', 'female', 'she', 'her']
    
    male_count = sum([1 for word in male_keywords if word.lower() in text.lower()])
    female_count = sum([1 for word in female_keywords if word.lower() in text.lower()])
    
    if male_count > female_count:
        demographics['cinsiyet'] = 'Erkek'
    elif female_count > male_count:
        demographics['cinsiyet'] = 'Kadın'
    else:
        demographics['cinsiyet'] = 'Bilinmiyor'
    
    # Eğitim seviyesi
    education_levels = {
        'Doktora': ['doktora', 'phd', 'ph.d', 'doktor', 'dr.'],
        'Yüksek Lisans': ['yüksek lisans', 'master', 'msc', 'm.sc', 'mba'],
        'Lisans': ['lisans', 'üniversite', 'fakülte', 'bsc', 'b.sc', 'bachelor'],
        'Ön Lisans': ['ön lisans', 'yüksekokul', 'meslek yüksekokul', 'associate'],
        'Lise': ['lise', 'high school', 'orta öğretim']
    }
    
    for level, keywords in education_levels.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            demographics['egitim'] = level
            break
    else:
        demographics['egitim'] = 'Bilinmiyor'
    
    # Yaş/deneyim tespiti
    experience_match = re.search(r'(\d+)\s*(?:yıl|year|sene)(?:\s+deneyim|experience)?', text.lower())
    if experience_match:
        experience = int(experience_match.group(1))
        demographics['deneyim_yil'] = experience
    else:
        demographics['deneyim_yil'] = 0
    
    # Yaş tespiti
    age_match = re.search(r'(?:yaş|age|years old)\s*:?\s*(\d+)', text.lower())
    if age_match:
        age = int(age_match.group(1))
        if 20 <= age <= 25:
            demographics['yas_grubu'] = '20-25'
        elif 26 <= age <= 30:
            demographics['yas_grubu'] = '26-30'
        elif 31 <= age <= 35:
            demographics['yas_grubu'] = '31-35'
        elif 36 <= age <= 40:
            demographics['yas_grubu'] = '36-40'
        elif 41 <= age <= 50:
            demographics['yas_grubu'] = '41-50'
        elif age > 50:
            demographics['yas_grubu'] = '51+'
        else:
            demographics['yas_grubu'] = 'Bilinmiyor'
    else:
        # Deneyime göre yaş tahmini
        if demographics['deneyim_yil'] > 0:
            est_age = demographics['deneyim_yil'] + 22  # Üniversite mezuniyeti + deneyim
            if 20 <= est_age <= 25:
                demographics['yas_grubu'] = '20-25'
            elif 26 <= est_age <= 30:
                demographics['yas_grubu'] = '26-30'
            elif 31 <= est_age <= 35:
                demographics['yas_grubu'] = '31-35'
            elif 36 <= est_age <= 40:
                demographics['yas_grubu'] = '36-40'
            elif 41 <= est_age <= 50:
                demographics['yas_grubu'] = '41-50'
            elif est_age > 50:
                demographics['yas_grubu'] = '51+'
            else:
                demographics['yas_grubu'] = 'Bilinmiyor'
        else:
            demographics['yas_grubu'] = 'Bilinmiyor'
    
    # Dil yetenekleri
    language_keywords = ['ingilizce', 'almanca', 'fransızca', 'ispanyolca', 'italyanca', 'rusça', 'çince', 'japonca',
                        'english', 'german', 'french', 'spanish', 'italian', 'russian', 'chinese', 'japanese']
    language_count = sum([1 for word in language_keywords if word.lower() in text.lower()])
    demographics['dil_sayisi'] = min(max(language_count, 1), 5)  # En az 1, en fazla 5 dil
    
    return demographics

def calculate_cv_score(demographics):
    """Demografik bilgilere göre CV skoru hesaplar"""
    cv_score = 50  # Başlangıç puanı
    
    # Eğitim puanı
    if demographics['egitim'] == 'Doktora':
        cv_score += 20
    elif demographics['egitim'] == 'Yüksek Lisans':
        cv_score += 15
    elif demographics['egitim'] == 'Lisans':
        cv_score += 10
    elif demographics['egitim'] == 'Ön Lisans':
        cv_score += 5
    
    # Deneyim puanı
    cv_score += min(demographics['deneyim_yil'] * 2, 30)
    
    # Dil puanı
    cv_score += demographics['dil_sayisi'] * 3
    
    # Yaşa göre (genç yaş grubuna bonus - ageism)
    if demographics['yas_grubu'] in ['20-25', '26-30', '31-35']:
        cv_score += 5
    elif demographics['yas_grubu'] in ['41-50', '51+']:
        cv_score -= 5
    
    # Cinsiyete göre eşit puan ekleme (sektör bias'ını temsil etmek için)
    # Gerçek uygulamada bu tür bias'lar ortadan kaldırılmalıdır
    if demographics['cinsiyet'] == 'Erkek':
        cv_score += 3  # Erkeklere teknoloji sektöründeki tarihi bias'ı simüle etmek için
    
    # Skor sınırlandırması
    cv_score = max(min(cv_score, 100), 0)
    
    return round(cv_score, 2)

def create_sample_cv_data(num_records=1000, seed=42):
    """Örnek CV verileri oluşturur"""
    np.random.seed(seed)
    
    # Demografik bilgiler
    genders = ['Erkek', 'Kadın']
    age_groups = ['20-25', '26-30', '31-35', '36-40', '41-50', '51+']
    education_levels = ['Lise', 'Ön Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora']
    positions = ['Yazılım Geliştirici', 'Veri Bilimci', 'Pazarlama Uzmanı', 'Satış Temsilcisi', 'İnsan Kaynakları']
    
    # Bias ekleyelim - işe alım oranlarında
    gender_bias = {'Erkek': 0.60, 'Kadın': 0.40}  # Erkekler lehine
    age_bias = {'20-25': 0.55, '26-30': 0.65, '31-35': 0.60, '36-40': 0.50, '41-50': 0.40, '51+': 0.30}  # Genç adaylar lehine
    
    data = []
    for _ in range(num_records):
        gender = np.random.choice(genders, p=[0.6, 0.4])  # Cinsiyet dağılımı
        age_group = np.random.choice(age_groups, p=[0.15, 0.25, 0.25, 0.15, 0.15, 0.05])  # Yaş dağılımı
        education = np.random.choice(education_levels, p=[0.1, 0.15, 0.45, 0.25, 0.05])  # Eğitim dağılımı
        position = np.random.choice(positions, p=[0.3, 0.2, 0.2, 0.15, 0.15])  # Pozisyon dağılımı
        
        # Özellik değerleri
        years_experience = np.random.randint(0, 20)
        languages = np.random.randint(1, 5)
        
        # Örnek AI tarafından hesaplanan skor (içinde bias var)
        base_score = np.random.normal(70, 15)
        
        # Bias faktörlerini ekleyelim
        if gender == 'Erkek':
            base_score += np.random.normal(5, 2)  # Erkeklere bonus
        
        if age_group in ['20-25', '26-30', '31-35']:
            base_score += np.random.normal(4, 2)  # Gençlere bonus
        elif age_group in ['41-50', '51+']:
            base_score += np.random.normal(-5, 2)  # Yaşlılara penaltı
        
        # Eğitim düzeyine göre bonus
        if education == 'Lisans':
            base_score += np.random.normal(3, 1)
        elif education == 'Yüksek Lisans':
            base_score += np.random.normal(5, 1)
        elif education == 'Doktora':
            base_score += np.random.normal(7, 1)
            
        # Skor sınırlandırması
        cv_score = max(min(base_score, 100), 0)
        
        # İşe alım kararı (bias içerir)
        hire_prob = (cv_score / 100) * gender_bias.get(gender, 0.5) * age_bias.get(age_group, 0.5)
        hired = 1 if np.random.random() < hire_prob else 0
            
        data.append({
            'cinsiyet': gender,
            'yas_grubu': age_group,
            'egitim': education,
            'pozisyon': position,
            'deneyim_yil': years_experience,
            'dil_sayisi': languages,
            'cv_skoru': round(cv_score, 2),
            'ise_alindi': hired
        })
    
    return pd.DataFrame(data)

def calculate_disparate_impact(df, protected_attribute, target_col):
    """Bias metriklerini hesaplar ve rapor oluşturur"""
    group_means = df.groupby(protected_attribute)[target_col].mean().reset_index()
    
    # Referans grup (ilk grup)
    reference_group = group_means[protected_attribute].iloc[0]
    reference_rate = group_means[target_col].iloc[0]
    
    # Bias metrikleri hesapla
    disparate_impacts = []
    for i, row in group_means.iterrows():
        group = row[protected_attribute]
        rate = row[target_col]
        
        if i == 0:  # Referans grup
            di = 1.0
            spd = 0.0
        else:  # Diğer gruplar
            di = rate / reference_rate if reference_rate > 0 else float('inf')
            spd = rate - reference_rate  # Statistical Parity Difference
        
        eod = 0.0  # Equal Opportunity Difference - Gelişmiş analizde hesaplanır
        
        disparate_impacts.append({
            'Grup': group,
            'İşe Alım Oranı': rate,
            'Disparate Impact': di,
            'Statistical Parity Diff': spd,
            'Equal Opportunity Diff': eod,
            'Bias Tespit Edildi': (di < 0.8 or di > 1.2)
        })
    
    return pd.DataFrame(disparate_impacts)

def calculate_fairness_metrics(df, protected_attribute, target_col, model=None, X=None, y=None):
    """AIF360 kullanarak gelişmiş fairness metrikleri hesaplar"""
    if not aif360_available:
        return None, "AIF360 kütüphanesi yüklenmemiş. pip install aif360 komutuyla yükleyebilirsiniz."
    
    try:
        # Korunan özellik ve hedef değişken için binary dataset oluştur
        privileged_groups = [{protected_attribute: df[protected_attribute].value_counts().index[0]}]
        unprivileged_groups = [{protected_attribute: df[protected_attribute].value_counts().index[1]}]
        
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[target_col],
            protected_attribute_names=[protected_attribute],
            privileged_protected_attributes=[1]
        )
        
        # Bias metrikleri hesapla
        metrics = BinaryLabelDatasetMetric(dataset, 
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        
        disparate_impact = metrics.disparate_impact()
        statistical_parity_difference = metrics.statistical_parity_difference()
        
        # Eğer model varsa, sınıflandırma metriklerini hesapla
        classification_metrics = {}
        if model is not None and X is not None and y is not None:
            # Model predictions
            y_pred = model.predict(X)
            
            # Korunan gruplara göre doğruluk
            accuracy_privileged = accuracy_score(
                y[df[protected_attribute] == privileged_groups[0][protected_attribute]], 
                y_pred[df[protected_attribute] == privileged_groups[0][protected_attribute]]
            )
            
            accuracy_unprivileged = accuracy_score(
                y[df[protected_attribute] == unprivileged_groups[0][protected_attribute]], 
                y_pred[df[protected_attribute] == unprivileged_groups[0][protected_attribute]]
            )
            
            classification_metrics = {
                'accuracy_privileged': accuracy_privileged,
                'accuracy_unprivileged': accuracy_unprivileged,
                'accuracy_difference': accuracy_privileged - accuracy_unprivileged
            }
        
        fairness_metrics = {
            'disparate_impact': disparate_impact,
            'statistical_parity_difference': statistical_parity_difference,
            'classification_metrics': classification_metrics
        }
        
        return fairness_metrics, None
    
    except Exception as e:
        return None, f"Fairness metrikleri hesaplanırken hata oluştu: {str(e)}"

def apply_bias_mitigation(df, protected_attribute, target_col, method="reweighing"):
    """AIF360 kullanarak bias azaltma algoritmaları uygular"""
    if not aif360_available:
        return None, "AIF360 kütüphanesi yüklenmemiş. pip install aif360 komutuyla yükleyebilirsiniz."
    
    try:
        # Korunan özellik ve hedef değişken için binary dataset oluştur
        privileged_groups = [{protected_attribute: df[protected_attribute].value_counts().index[0]}]
        unprivileged_groups = [{protected_attribute: df[protected_attribute].value_counts().index[1]}]
        
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[target_col],
            protected_attribute_names=[protected_attribute],
            privileged_protected_attributes=[1]
        )
        
        # Bias azaltma algoritması seç ve uygula
        if method == "reweighing":
            mitigator = Reweighing(unprivileged_groups=unprivileged_groups,
                                privileged_groups=privileged_groups)
            transformed_dataset = mitigator.fit_transform(dataset)
        
        elif method == "disparate_impact_remover":
            mitigator = DisparateImpactRemover(repair_level=0.8)
            transformed_dataset = mitigator.fit_transform(dataset)
        
        else:
            return None, f"Desteklenmeyen bias azaltma yöntemi: {method}"
        
        # Dönüştürülmüş veriyi DataFrame'e çevir
        transformed_df = transformed_dataset.convert_to_dataframe()[0]
        
        return transformed_df, None
    
    except Exception as e:
        return None, f"Bias azaltma uygulanırken hata oluştu: {str(e)}"

def train_model(df, protected_attribute, target_col):
    """CV verileri üzerinde model eğitir ve model performans metriklerini hesaplar"""
    # Eğitim ve test verilerini ayır
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Kategorik değişkenleri one-hot encoding
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model eğitimi
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model performansını değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Korunan özelliğe göre model performansı
    if protected_attribute in df.columns:
        group_performance = {}
        protected_col_onehot = [col for col in X.columns if protected_attribute in col]
        
        if len(protected_col_onehot) > 0:
            # One-hot encoded sütunlardan orijinal korunan özellik değerlerini geri çıkarmak
            # için X_test'e bakıp grupları belirlemek gerekir - basitleştirmek için burada atlanıyor
            pass
        
        # Gruplara göre performans ölçümleri
        for group in df[protected_attribute].unique():
            group_mask = df.loc[X_test.index, protected_attribute] == group
            if sum(group_mask) > 0:
                group_y_test = y_test[group_mask]
                group_y_pred = y_pred[group_mask]
                group_performance[group] = {
                    'accuracy': accuracy_score(group_y_test, group_y_pred),
                    'precision': sum((group_y_pred == 1) & (group_y_test == 1)) / sum(group_y_pred == 1) if sum(group_y_pred == 1) > 0 else 0,
                    'recall': sum((group_y_pred == 1) & (group_y_test == 1)) / sum(group_y_test == 1) if sum(group_y_test == 1) > 0 else 0
                }
    
    return model, X_test, y_test, y_pred, accuracy, cm, report, group_performance

def analyze_feature_importance(model, X):
    """Model özellik önemini analiz eder"""
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def generate_shap_values(model, X):
    """SHAP değerlerini hesaplar (açıklanabilirlik)"""
    if not explainability_available:
        return None, "SHAP kütüphanesi yüklenmemiş. pip install shap komutuyla yükleyebilirsiniz."
    
    try:
        # SHAP değerleri hesapla
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        return shap_values, explainer, None
    
    except Exception as e:
        return None, None, f"SHAP değerleri hesaplanırken hata oluştu: {str(e)}"

def prepare_lime_explainer(X_train, feature_names):
    """LIME açıklayıcı oluşturur"""
    if not explainability_available:
        return None, "LIME kütüphanesi yüklenmemiş. pip install lime komutuyla yükleyebilirsiniz."
    
    try:
        # LIME açıklayıcı oluştur
        explainer = LimeTabularExplainer(
            X_train.values, 
            feature_names=feature_names, 
            class_names=['Reddedildi', 'İşe Alındı'],
            mode='classification'
        )
        
        return explainer, None
    
    except Exception as e:
        return None, f"LIME açıklayıcı oluşturulurken hata oluştu: {str(e)}"

# ---- Ana Uygulama ----
# Sidebar
st.sidebar.header("📊 Analiz Parametreleri")

# Veri kaynağı seçenekleri
data_option = st.sidebar.radio("Veri kaynağı:", ["CV Yükle", "Örnek Veri"])

# Dosya yükleme
if data_option == "CV Yükle":
    # PDF/DOCX destek kontrolü
    if not pdf_docx_support:
        st.sidebar.warning("PDF ve DOCX destekleri eksik. pip install PyPDF2 python-docx")
        file_types = ["csv"]
        accept_type = "CSV dosyanızı yükleyin"
    else:
        file_types = ["csv", "pdf", "docx", "doc", "txt"]
        accept_type = "CV dosyalarınızı yükleyin"
    
    uploaded_files = st.sidebar.file_uploader(accept_type, type=file_types, accept_multiple_files=True)

# Örnek veri parametreleri
if data_option == "Örnek Veri" or (data_option == "CV Yükle" and not uploaded_files):
    sample_size = st.sidebar.slider("Örnek veri sayısı", 100, 5000, 1000, 100)
    if data_option == "CV Yükle" and not uploaded_files:
        st.sidebar.info("Dosya yüklenmedi. Örnek veri kullanılacak.")

# Analiz parametreleri
st.sidebar.subheader("Korunan Özellik ve Hedef")
protected_attribute = st.sidebar.selectbox(
    "Korunan özelliği seçin", 
    ["cinsiyet", "yas_grubu", "egitim"],
    index=0
)

target_col = st.sidebar.selectbox(
    "Hedef değişkeni seçin",
    ["ise_alindi"],
    index=0
)

# Bias azaltma yöntemi (AIF360 varsa)
st.sidebar.subheader("Bias Azaltma")
if aif360_available:
    apply_mitigation = st.sidebar.checkbox("Bias azaltma uygula", value=False)
    if apply_mitigation:
        mitigation_method = st.sidebar.selectbox(
            "Bias azaltma yöntemi",
            ["reweighing", "disparate_impact_remover"],
            index=0
        )
else:
    apply_mitigation = False
    st.sidebar.warning("Bias azaltma için AIF360 gerekli")

# Analiz butonu
analyze_button = st.sidebar.button("Analiz Et", type="primary")

# Model ve açıklanabilirlik ayarları
st.sidebar.subheader("Model Analizi")
model_type = st.sidebar.selectbox(
    "Model türü",
    ["Random Forest", "Lojistik Regresyon"],
    index=0
)

if explainability_available:
    st.sidebar.subheader("Açıklanabilirlik")
    explainability_method = st.sidebar.multiselect(
        "Açıklanabilirlik yöntemleri",
        ["SHAP", "LIME", "ELI5"],
        default=["SHAP"]
    )
else:
    explainability_method = []
    st.sidebar.warning("Açıklanabilirlik için SHAP, LIME ve ELI5 gerekli")

# Sekmeleri oluştur
tabs = st.tabs(["Veri Analizi", "Bias Analizi", "Model Analizi", "Açıklanabilirlik", "EU AI Act Rehberi", "Deployment"])

# Veri Yükleme veya Oluşturma
if data_option == "CV Yükle" and uploaded_files:
    cv_data = []
    
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            # CSV dosyasını doğrudan oku
            df = pd.read_csv(uploaded_file)
            st.success(f"CSV dosyası başarıyla yüklendi: {uploaded_file.name}")
            break  # CSV dosyası bulunduğunda diğer dosyaları işlemeyi durdur
        
        elif pdf_docx_support:
            # PDF, DOCX ve TXT dosyaları için metin çıkarma
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            text_content = ""
            try:
                if file_ext == 'pdf':
                    with open(tmp_path, 'rb') as f:
                        text_content = extract_text_from_pdf(f)
                elif file_ext in ['docx', 'doc']:
                    with open(tmp_path, 'rb') as f:
                        text_content = extract_text_from_docx(f)
                elif file_ext == 'txt':
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        
                # Demografik özellikleri çıkar
                demographics = extract_demographics(text_content)
                
                # CV skoru hesapla
                cv_score = calculate_cv_score(demographics)
                
                # İşe alım kararı (örnek)
                hired = 1 if cv_score > 65 else 0
                
                # Veri dizisine ekle
                cv_data.append({
                    'dosya_adi': uploaded_file.name,
                    'cinsiyet': demographics['cinsiyet'],
                    'yas_grubu': demographics['yas_grubu'],
                    'egitim': demographics['egitim'],
                    'deneyim_yil': demographics['deneyim_yil'],
                    'dil_sayisi': demographics.get('dil_sayisi', 1),
                    'cv_skoru': cv_score,
                    'ise_alindi': hired
                })
                
                st.success(f"Dosya işlendi: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Dosya işleme hatası ({uploaded_file.name}): {str(e)}")
            
            finally:
                # Geçici dosyayı temizle
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    if cv_data:  # Eğer CV'lerden veri çıkarıldıysa
        df = pd.DataFrame(cv_data)
    elif 'df' not in locals():  # Hiç CSV yüklenmemiş ve CV'lerden veri çıkarılamamışsa
        st.warning("Hiçbir dosya işlenemedi. Örnek veri kullanılacak.")
        df = create_sample_cv_data(num_records=sample_size)

else:  # Örnek veri kullan
    df = create_sample_cv_data(num_records=sample_size)

# Veri Analizi Sekmesi
with tabs[0]:
    st.header("📊 Veri Analizi")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Veri Özeti")
        df_info = pd.DataFrame({
            'Sütun': df.columns,
            'Veri Tipi': df.dtypes,
            'Boş Değer': df.isnull().sum(),
            'Benzersiz Değer': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(df_info)
        
        st.subheader("Veri Önizleme")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Temel İstatistikler")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        st.dataframe(df[numeric_cols].describe())
        
        st.subheader("Korelasyon Matrisi")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    # Demografik dağılımlar
    st.subheader("Demografik Dağılımlar")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Cinsiyet dağılımı
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='cinsiyet', data=df, ax=ax)
        ax.set_title('Cinsiyet Dağılımı')
        st.pyplot(fig)
        
        # Eğitim dağılımı
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='egitim', data=df, ax=ax)
        ax.set_title('Eğitim Seviyesi Dağılımı')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Yaş grubu dağılımı
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='yas_grubu', data=df, ax=ax)
        ax.set_title('Yaş Grubu Dağılımı')
        st.pyplot(fig)
        
        # İşe alım oranları
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='ise_alindi', data=df, ax=ax)
        ax.set_title('İşe Alım Dağılımı')
        ax.set_xticklabels(['Reddedilen', 'İşe Alınan'])
        st.pyplot(fig)
    
    # CV skoru dağılımı
    st.subheader("CV Skoru Dağılımı")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['cv_skoru'], bins=20, kde=True, ax=ax)
    ax.set_title('CV Skoru Dağılımı')
    ax.set_xlabel('CV Skoru')
    ax.set_ylabel('Frekans')
    st.pyplot(fig)

# Bias Analizi Sekmesi
if analyze_button:
    with tabs[1]:
        st.header("⚖️ Bias Analizi")
        
        # Demografik Dağılım
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Demografik Dağılım")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=protected_attribute, hue=target_col, data=df, ax=ax)
            ax.set_title(f"{protected_attribute.capitalize()} Demografik Dağılımı")
            ax.set_xlabel(protected_attribute.capitalize())
            ax.set_ylabel("Kişi Sayısı")
            if target_col == 'ise_alindi':
                ax.legend(['Reddedilen', 'İşe Alınan'])
            st.pyplot(fig)
        
        with col2:
            st.subheader("İşe Alım Oranları")
            hire_rates = df.groupby(protected_attribute)[target_col].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=protected_attribute, y=target_col, data=hire_rates, ax=ax)
            ax.set_title(f"{protected_attribute.capitalize()}'e Göre İşe Alım Oranları")
            ax.set_xlabel(protected_attribute.capitalize())
            ax.set_ylabel("İşe Alım Oranı")
            for i, v in enumerate(hire_rates[target_col]):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            st.pyplot(fig)
        
        # Disparate Impact analizi
        st.subheader("Disparate Impact Analizi")
        
        di_results = calculate_disparate_impact(df, protected_attribute, target_col)
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            # Sonuçları renklendirerek göster
            def highlight_bias(val):
                if val == True:
                    return 'background-color: #ffcccb'  # Light red
                else:
                    return ''
            
            st.dataframe(di_results.style.applymap(highlight_bias, subset=['Bias Tespit Edildi']))
            
            # Bias bilgilendirmesi
            if di_results['Bias Tespit Edildi'].any():
                st.error("⚠️ Bias tespit edildi! EU AI Act kapsamında düzeltici önlemler gerekli.")
            else:
                st.success("✅ Bias tespit edilmedi. Değerler kabul edilebilir aralıkta.")
        
        with col4:
            # Disparate Impact Grafiği
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Grup', y='Disparate Impact', data=di_results, ax=ax)
            
            # 0.8 ve 1.2 sınırlarını göster
            ax.axhline(y=0.8, color='r', linestyle='--', label='Alt Sınır (0.8)')
            ax.axhline(y=1.2, color='r', linestyle='--', label='Üst Sınır (1.2)')
            ax.axhline(y=1.0, color='g', linestyle='-', label='Eşitlik (1.0)')
            
            ax.set_title("Disparate Impact Analizi")
            ax.set_xlabel(protected_attribute.capitalize())
            ax.set_ylabel("Disparate Impact")
            ax.legend()
            st.pyplot(fig)
        
        # CV Skorları Analizi
        st.subheader("CV Skorları Dağılımı")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=protected_attribute, y='cv_skoru', data=df, ax=ax)
        ax.set_title(f"{protected_attribute.capitalize()} Gruplarına Göre CV Skorları")
        ax.set_xlabel(protected_attribute.capitalize())
        ax.set_ylabel("CV Skoru")
        st.pyplot(fig)
        
        # IBM AI Fairness 360 Metrikleri (eğer mevcut ise)
        st.subheader("IBM AI Fairness 360 Metrikleri")
        
        if aif360_available:
            fairness_metrics, error_msg = calculate_fairness_metrics(df, protected_attribute, target_col)
            
            if fairness_metrics:
                col5, col6 = st.columns([1, 1])
                
                with col5:
                    st.info(f"""
                    **Disparate Impact:** {fairness_metrics['disparate_impact']:.4f}  
                    (Hedef: 0.8 - 1.2 arası)
                    
                    **Statistical Parity Difference:** {fairness_metrics['statistical_parity_difference']:.4f}  
                    (Hedef: 0'a yakın değer)
                    """)
                
                with col6:
                    if fairness_metrics['classification_metrics']:
                        cm = fairness_metrics['classification_metrics']
                        st.info(f"""
                        **Ayrıcalıklı Grup Doğruluğu:** {cm['accuracy_privileged']:.4f}
                        
                        **Dezavantajlı Grup Doğruluğu:** {cm['accuracy_unprivileged']:.4f}
                        
                        **Doğruluk Farkı:** {cm['accuracy_difference']:.4f}
                        """)
            else:
                st.warning(error_msg)
        else:
            st.warning("IBM AI Fairness 360 kütüphanesi yüklenmemiş. Gelişmiş metrikler için AIF360 gerekli.")
        
        # Bias Azaltma (eğer seçilmişse)
        if apply_mitigation and aif360_available:
            st.subheader("Bias Azaltma Sonuçları")
            
            transformed_df, error_msg = apply_bias_mitigation(df, protected_attribute, target_col, method=mitigation_method)
            
            if transformed_df is not None:
                col7, col8 = st.columns([1, 1])
                
                with col7:
                    st.subheader("Azaltma Öncesi İşe Alım Oranları")
                    before_rates = df.groupby(protected_attribute)[target_col].mean().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=protected_attribute, y=target_col, data=before_rates, ax=ax)
                    ax.set_title("Azaltma Öncesi")
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                
                with col8:
                    st.subheader("Azaltma Sonrası İşe Alım Oranları")
                    after_rates = transformed_df.groupby(protected_attribute)[target_col].mean().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=protected_attribute, y=target_col, data=after_rates, ax=ax)
                    ax.set_title(f"Azaltma Sonrası ({mitigation_method})")
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                
                # Bias metrikleri değişimi
                before_di = calculate_disparate_impact(df, protected_attribute, target_col)
                after_di = calculate_disparate_impact(transformed_df, protected_attribute, target_col)
                
                st.subheader("Bias Metriklerindeki Değişim")
                
                col9, col10 = st.columns([1, 1])
                
                with col9:
                    st.markdown("**Azaltma Öncesi Disparate Impact:**")
                    st.dataframe(before_di[['Grup', 'Disparate Impact', 'Bias Tespit Edildi']])
                
                with col10:
                    st.markdown("**Azaltma Sonrası Disparate Impact:**")
                    st.dataframe(after_di[['Grup', 'Disparate Impact', 'Bias Tespit Edildi']])
                
                # Veri setini güncelle
                st.success(f"Bias azaltma başarıyla uygulandı. İşe alım oranlarındaki fark azaltıldı.")
                st.info("Sonraki analizler bias azaltma uygulanmış veri üzerinde yapılacak.")
                df = transformed_df  # Veri setini güncelle
            else:
                st.error(error_msg)
        
        # Bias Azaltma Önerileri
        st.subheader("Önerilen Düzeltici Önlemler")
        
        if di_results['Bias Tespit Edildi'].any():
            st.markdown("""
            ### Yapılabilecek İyileştirmeler:
            
            1. **Veri Toplama ve İşlemede Düzeltmeler**:
               - CV'lerden demografik bilgileri çıkarın (cinsiyet, yaş, resim)
               - İsim bilgisini anonimleştirin
               - Bias kontrollü veri toplama süreçleri oluşturun
            
            2. **Model Geliştirmede Bias Azaltma**:
               - Pre-processing: Reweighing, Disparate Impact Remover
               - In-processing: Adversarial Debiasing, Prejudice Remover
               - Post-processing: Calibrated Equalized Odds, Reject Option Classification
            
            3. **İnsan Gözetimi ve Süreç İyileştirmeleri**:
               - Eğitimli kişiler tarafından nihai kontrol
               - Çeşitlilik ekibi tarafından düzenli bias denetimleri
               - Adil işe alım süreçleri hakkında eğitim
            """)
        else:
            st.info("Disparate Impact değerleri kabul edilebilir aralıkta. Yine de düzenli bias denetimleri yapmanız önerilir.")

    # Model Analizi Sekmesi
    with tabs[2]:
        st.header("🤖 Model Analizi")
        
        # Model eğitimi ve değerlendirme
        st.subheader("Model Performansı")
        
        try:
            # Model eğitim ve değerlendirme
            model, X_test, y_test, y_pred, accuracy, cm, report, group_performance = train_model(df, protected_attribute, target_col)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.info(f"""
                **Model Türü**: {model_type}
                **Genel Doğruluk**: {accuracy:.4f}
                **Precision**: {report['1']['precision']:.4f}
                **Recall**: {report['1']['recall']:.4f}
                **F1 Skoru**: {report['1']['f1-score']:.4f}
                """)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Tahmin')
                ax.set_ylabel('Gerçek')
                ax.set_xticklabels(['Reddedilen', 'İşe Alınan'])
                ax.set_yticklabels(['Reddedilen', 'İşe Alınan'])
                st.pyplot(fig)
            
            with col2:
                # Gruplar Arası Performans
                st.subheader("Demografik Gruplar Arası Performans")
                
                group_df = pd.DataFrame.from_dict(group_performance, orient='index')
                group_df = group_df.reset_index().rename(columns={'index': protected_attribute})
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=protected_attribute, y='accuracy', data=group_df, ax=ax)
                ax.set_title(f"{protected_attribute.capitalize()} Gruplarına Göre Model Doğruluğu")
                ax.set_xlabel(protected_attribute.capitalize())
                ax.set_ylabel("Doğruluk")
                for i, v in enumerate(group_df['accuracy']):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
                st.pyplot(fig)
            
            # Özellik Önemliliği
            st.subheader("Özellik Önemliliği")
            
            feature_importance = analyze_feature_importance(model, X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
            ax.set_title("En Önemli 10 Özellik")
            ax.set_xlabel("Önem Derecesi")
            ax.set_ylabel("Özellik")
            st.pyplot(fig)
            
            # Microsoft Fairlearn metrikleri (eğer mevcutsa)
            if fairlearn_available:
                st.subheader("Microsoft Fairlearn Metrikleri")
                
                try:
                    # Sensitive attribute
                    A = df[protected_attribute]
                    
                    # Calculate fairness metrics
                    dpd = demographic_parity_difference(y_true=y_test, 
                                                    y_pred=y_pred, 
                                                    sensitive_features=A.loc[X_test.index])
                    
                    eod = equalized_odds_difference(y_true=y_test, 
                                                y_pred=y_pred, 
                                                sensitive_features=A.loc[X_test.index])
                    
                    st.info(f"""
                    **Demographic Parity Difference**: {dpd:.4f}  
                    (0'a yakın olması daha adil)
                    
                    **Equalized Odds Difference**: {eod:.4f}  
                    (0'a yakın olması daha adil)
                    """)
                    
                except Exception as e:
                    st.warning(f"Fairlearn metrikleri hesaplanırken hata oluştu: {str(e)}")
            
            else:
                st.warning("Microsoft Fairlearn metrikleri için kütüphane yüklenmemiş.")
            
            # Model İyileştirme Önerileri
            st.subheader("Model İyileştirme Önerileri")
            
            # En büyük doğruluk farkı
            if len(group_performance) > 1:
                max_accuracy = max([gp['accuracy'] for gp in group_performance.values()])
                min_accuracy = min([gp['accuracy'] for gp in group_performance.values()])
                accuracy_diff = max_accuracy - min_accuracy
                
                if accuracy_diff > 0.1:  # %10'dan fazla fark varsa
                    st.warning(f"""
                    ⚠️ Gruplar arasında önemli performans farkı tespit edildi ({accuracy_diff:.2f}).
                    Bu durum, modelin bazı demografik gruplar için daha az doğru tahminler yaptığını gösterir.
                    """)
                    
                    st.markdown("""
                    ### İyileştirme Önerileri:
                    
                    1. **Veri Dengesi**:
                       - Dezavantajlı grup için daha fazla veri toplayın
                       - Veri augmentasyonu veya resampling yöntemleri kullanın
                    
                    2. **Özellik Mühendisliği**:
                       - Dezavantajlı grup için önemli özellikleri belirleyin
                       - Yeni, bias azaltıcı özellikler ekleyin
                    
                    3. **Model Seçimi ve Hiperparametre Optimizasyonu**:
                       - Farklı model türlerini deneyin
                       - Fairness kısıtları ile hiperparametre optimizasyonu yapın
                    
                    4. **Fairness Aware Öğrenme**:
                       - Adversarial Debiasing gibi fairness-aware modeller kullanın
                       - Post-processing yöntemleri ile model çıktılarını kalibre edin
                    """)
                else:
                    st.success(f"""
                    ✅ Gruplar arasındaki performans farkı kabul edilebilir düzeyde ({accuracy_diff:.2f}).
                    Model, farklı demografik gruplar için benzer doğrulukta tahminler yapıyor.
                    """)
        
        except Exception as e:
            st.error(f"Model analizi sırasında bir hata oluştu: {str(e)}")

    # Açıklanabilirlik Sekmesi
    with tabs[3]:
        st.header("🔍 Model Açıklanabilirliği")
        
        if not explainability_available:
            st.warning("Açıklanabilirlik analizleri için SHAP, LIME veya ELI5 kütüphaneleri gereklidir.")
        elif 'model' not in locals():
            st.warning("Açıklanabilirlik analizleri için önce Model Analizi sekmesinde model eğitimi yapılmalıdır.")
        else:
            # SHAP Değerleri
            if "SHAP" in explainability_method:
                st.subheader("SHAP Değerleri Analizi")
                
                try:
                    shap_values, explainer, error_msg = generate_shap_values(model, X_test)
                    
                    if shap_values is not None:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # SHAP özet grafiği
                            st.write("SHAP Özet Grafiği (Genel Özellik Etkileri)")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                            st.pyplot(fig)
                        
                        with col2:
                            # SHAP bağımlılık grafiği
                            st.write("SHAP Bağımlılık Grafiği (İşe Alım Tahmini)")
                            most_imp_feature = feature_importance['feature'].iloc[0]
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.dependence_plot(most_imp_feature, shap_values[1], X_test, show=False)
                            st.pyplot(fig)
                        
                        # SHAP Kararlar
                        st.subheader("SHAP Değerleri ile Kararların Açıklanması")
                        
                        sample_idx = np.random.choice(len(X_test))
                        sample_instance = X_test.iloc[sample_idx]
                        sample_prediction = model.predict([sample_instance])[0]
                        
                        st.write(f"""
                        **Örnek Aday**:
                        - Tahmin: {"İşe Alındı" if sample_prediction == 1 else "Reddedildi"}
                        - Gerçek: {"İşe Alındı" if y_test.iloc[sample_idx] == 1 else "Reddedildi"}
                        """)
                        
                        # Waterfall grafiği
                        st.write("Karar Açıklaması (Waterfall Grafiği)")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], 
                                                        shap_values[1][sample_idx], 
                                                        feature_names=X_test.columns,
                                                        show=False)
                        st.pyplot(fig)
                    else:
                        st.error(error_msg)
                
                except Exception as e:
                    st.error(f"SHAP analizi sırasında bir hata oluştu: {str(e)}")
            
            # LIME Açıklamaları
            if "LIME" in explainability_method:
                st.subheader("LIME Açıklamaları")
                
                try:
                    lime_explainer, error_msg = prepare_lime_explainer(X_test, X_test.columns)
                    
                    if lime_explainer is not None:
                        # Rastgele bir örnek seç
                        sample_idx = np.random.choice(len(X_test))
                        sample_instance = X_test.iloc[sample_idx]
                        prediction = model.predict_proba([sample_instance])[0]
                        
                        st.write(f"""
                        **Örnek Aday**:
                        - İşe Alınma Olasılığı: {prediction[1]:.2f}
                        - Tahmin: {"İşe Alındı" if prediction[1] > 0.5 else "Reddedildi"}
                        """)
                        
                        # LIME açıklaması oluştur
                        explanation = lime_explainer.explain_instance(
                            sample_instance.values, 
                            model.predict_proba,
                            num_features=10
                        )
                        
                        # Açıklamayı görselleştir
                        st.write("Karar Açıklaması (LIME)")
                        fig = explanation.as_pyplot_figure(label=1)
                        st.pyplot(fig)
                        
                        # LIME açıklamasını tablo olarak göster
                        st.write("Özellik Etkileri (LIME)")
                        feature_values = explanation.as_list()
                        feature_table = pd.DataFrame(feature_values, columns=['Özellik', 'Etki'])
                        st.dataframe(feature_table)
                    else:
                        st.error(error_msg)
                
                except Exception as e:
                    st.error(f"LIME analizi sırasında bir hata oluştu: {str(e)}")
            
            # ELI5 Açıklamaları
            if "ELI5" in explainability_method:
                st.subheader("ELI5 Açıklamaları")
                
                try:
                    # PermutationImportance ile özellik önemlerini hesapla
                    perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
                    
                    # ELI5 ile özellik önemlerini görselleştir
                    st.write("Özellik Önemleri (Permutation Importance)")
                    eli5_html = eli5.show_weights(perm, feature_names=X_test.columns.tolist())
                    st.components.v1.html(eli5_html.data, height=500)
                    
                    # Örnek tahmin açıklaması
                    st.write("Örnek Aday için Tahmin Açıklaması")
                    sample_idx = np.random.choice(len(X_test))
                    sample_instance = X_test.iloc[sample_idx]
                    
                    eli5_prediction = eli5.show_prediction(model, sample_instance, 
                                                        feature_names=X_test.columns.tolist())
                    st.components.v1.html(eli5_prediction.data, height=500)
                
                except Exception as e:
                    st.error(f"ELI5 analizi sırasında bir hata oluştu: {str(e)}")
            
            # Açıklanabilirlik Karşılaştırması
            if len(explainability_method) > 1:
                st.subheader("Açıklanabilirlik Yöntemlerinin Karşılaştırması")
                
                st.markdown("""
                ### Karşılaştırma Tablosu
                
                | Yöntem | Güçlü Yönleri | Zayıf Yönleri |
                |--------|--------------|--------------|
                | **SHAP** | - Teorik temellere dayanır (Shapley değerleri)<br>- Tutarlı ve kesin sonuçlar<br>- Küresel ve yerel açıklamalar | - Hesaplama maliyeti yüksek<br>- Karmaşık modellerde yavaş |
                | **LIME** | - Hızlı ve sezgisel<br>- Model-agnostik<br>- Yerel açıklamalar için ideal | - Örnekleme varyansı yüksek<br>- Kararsız sonuçlar üretebilir |
                | **ELI5** | - Kullanımı kolay<br>- Permütasyon önem skorları tutarlı<br>- Görsel açıklamalar | - Özellikler arası etkileşimleri yakalayamaz<br>- Genellikle küresel açıklamalar sağlar |
                
                ### Hangi Durumda Hangi Yöntem Kullanılmalı?
                
                - **SHAP**: Kesin sonuçlar ve teorik güvence istendiğinde
                - **LIME**: Hızlı, yerel açıklamalar gerektiğinde
                - **ELI5**: Temel düzeyde, kolay anlaşılır açıklamalar istendiğinde
                
                Bias tespiti ve analizi için birden fazla açıklanabilirlik yönteminin birlikte kullanılması en iyi sonucu verir.
                """)

    # EU AI Act Rehberi Sekmesi
    with tabs[4]:
        st.header("📜 EU AI Act Rehberi")
        
        st.markdown("""
        ## EU AI Act ve İşe Alım Sistemleri
        
        **AB Yapay Zeka Yasası (EU AI Act)**, Nisan 2024'te onaylanmış ve işe alım, performans değerlendirme ve insan kaynakları yönetimi için kullanılan AI sistemlerini **yüksek riskli AI uygulamaları** olarak sınıflandırmaktadır (Madde 6 ve Ek III, Madde 4).
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*Eo4IFEtZWBh9jOA_HT-6_g.png", 
                    caption="EU AI Act Yüksek Riskli AI Sistemleri Sınıflandırması")
        
        with col2:
            st.markdown("""
            ### EU AI Act Riskli AI Sistemi Kategorileri:
            
            1. **Yasaklanmış AI Sistemleri**:
               - Bilinçaltı manipülasyon
               - Hassas grupların istismarı
               - Sosyal skor sistemleri
               - Gerçek zamanlı biyometrik tanımlama (istisnalar hariç)
            
            2. **Yüksek Riskli AI Sistemleri** (İşe alım sistemleri bu kategoridedir):
               - Kritik altyapı
               - Eğitim ve mesleki eğitim
               - İstihdam, işçi yönetimi, öz istihdama erişim
               - Temel özel ve kamu hizmetlerine erişim
               - Kolluk kuvvetleri
               - Göç, iltica ve sınır kontrolü
               - Adalet ve demokratik süreçler
            
            3. **Sınırlı Riskli AI Sistemleri**:
               - Chatbotlar
               - Duygu tanıma sistemleri
               - Biyometrik kategorizasyon
               - Sentetik içerik oluşturma (deepfakes)
            
            4. **Minimal Riskli AI Sistemleri**:
               - AI destekli video oyunları
               - Spam filtreleri
               - Diğer düşük riskli uygulamalar
            """)
        
        st.subheader("İşe Alım Sistemleri için Uyumluluk Gereksinimleri")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("""
            ### 1. Risk Yönetim Sistemi
            
            - Sistematik risk tanımlama ve değerlendirme
            - Risk azaltma önlemleri geliştirme
            - Düzenli risk izleme ve güncelleme
            - Bias ve ayrımcılık risklerini belirleme
            
            ### 2. Veri ve Veri Yönetişimi
            
            - Yüksek kaliteli eğitim, doğrulama ve test verileri
            - Uygun veri yönetişimi uygulamaları
            - Veri setlerinin uygun, temsili ve bias içermediğini sağlama
            - Veri ön işleme, işleme ve değerlendirme süreçleri
            
            ### 3. Teknik Dokümantasyon
            
            - Sistem tasarımı ve mimarisi
            - Algoritma özellikleri ve karar mekanizması
            - Eğitim metodolojisi ve veri seti özellikleri
            - Model kartları ve performans metrikleri
            - Bias ölçüm ve azaltma süreçleri
            """)
        
        with col4:
            st.markdown("""
            ### 4. Kayıt Tutma ve Şeffaflık
            
            - Otomatik kayıt tutma (log)
            - Sistemin işleyişini izleme
            - Beklenmeyen sonuçları tespit etme
            - Audit trail oluşturma
            
            ### 5. İnsan Gözetimi
            
            - İnsan tarafından etkili gözetim
            - Son kararı insan tarafından onaylama
            - AI sistemini geçersiz kılma yetkisi
            - Sonuçları yorumlama ve müdahale etme kapasitesi
            
            ### 6. Doğruluk, Dayanıklılık ve Güvenlik
            
            - Kabul edilebilir doğruluk seviyesi
            - Farklı demografik gruplar için eşit performans
            - Hatalara, tutarsızlıklara ve siber saldırılara karşı dayanıklılık
            - Yedekleme ve güvenlik planları
            """)
        
        st.subheader("Bias Değerlendirme ve Azaltma")
        
        st.markdown("""
        ### EU AI Act Kapsamında Bias Ölçüm Metrikleri
        
        | Metrik | Açıklama | Kabul Edilebilir Aralık |
        |--------|-----------|------------------------|
        | **Disparate Impact** | Dezavantajlı/Avantajlı grup seçilme oranı | 0.8 - 1.2 |
        | **Statistical Parity Difference** | Gruplar arası seçilme oranı farkı | -0.1 - 0.1 |
        | **Equal Opportunity Difference** | Gruplar arası doğru pozitif oranı farkı | -0.1 - 0.1 |
        | **Average Odds Difference** | Gruplar arası FPR ve TPR ortalaması farkı | -0.1 - 0.1 |
        | **Theil Index** | Tahminlerdeki eşitsizlik ölçümü | 0 - 0.2 |
        
        ### Bias Azaltma Stratejileri
        
        1. **Pre-processing (İşlem Öncesi)**
           - Veri Resampling
           - Disparate Impact Remover
           - Learning Fair Representations
           - Optimized Preprocessing
        
        2. **In-processing (İşlem Sırasında)**
           - Adversarial Debiasing
           - Exponentiated Gradient Reduction
           - Grid Search Reduction
           - Prejudice Remover
        
        3. **Post-processing (İşlem Sonrası)**
           - Equalized Odds Postprocessing
           - Calibrated Equalized Odds
           - Reject Option Classification
        """)
        
        st.subheader("Cezalar ve Uyumsuzluk Riskleri")
        
        st.info("""
        ### Ceza ve Yaptırımlar
        
        EU AI Act, uyumsuzluk durumunda ciddi yaptırımlar öngörmektedir:
        
        - **Yasaklanmış AI Sistemleri**: 35 milyon € veya global cironun %7'sine kadar
        - **Yüksek Riskli AI Sistemlerine Uyumsuzluk**: 15 milyon € veya global cironun %3'üne kadar
        - **Diğer Uyumsuzluklar**: 7.5 milyon € veya global cironun %1.5'ine kadar
        
        Ayrıca:
        - AI sistemlerinin piyasadan çekilmesi
        - Faaliyet kısıtlamaları 
        - Kamuya açık uyarılar
        - Operasyonel lisans iptali
        """)
        
        st.subheader("Uyumluluk Kontrol Listesi")
        
        uyumluluk_listesi = [
            "Risk değerlendirmesi yaptınız mı?",
            "Veri setinizdeki potansiyel bias'ları değerlendirdiniz mi?",
            "Farklı demografik gruplar için model performansını ölçtünüz mü?",
            "Dokümantasyon ve model kartları hazırladınız mı?",
            "İnsan gözetimi mekanizması tasarladınız mı?",
            "Bias azaltma stratejileri uyguladınız mı?",
            "Düzenli sistem denetimi için prosedürler belirlediniz mi?",
            "Sistemin şeffaflığı ve açıklanabilirliği sağlandı mı?",
            "Kullanıcı/çalışan bilgilendirme süreçleri oluşturuldu mu?",
            "Olay yanıt/müdahale planı geliştirildi mi?"
        ]
        
        for i, item in enumerate(uyumluluk_listesi):
            checked = st.checkbox(item, key=f"compliance_{i}")
            if not checked:
                st.warning(f"⚠️ {item} - EU AI Act uyumluluğu için gerekli")
        
        # Bitirme notu
        st.info("""
        **Not**: Bu rehber genel bilgilendirme amaçlıdır. Gerçek bir EU AI Act uyumluluk süreci için hukuk danışmanları 
        ve AI etik uzmanlarından destek alınması önerilir.
        """)

    # Deployment Sekmesi
    with tabs[5]:
        st.header("🚀 Deployment Rehberi")
        
        st.markdown("""
        ## CV Bias Analiz Aracının Deployment Seçenekleri
        
        Bu CV Bias Analiz aracını farklı ortamlarda nasıl deploy edebileceğinize dair yönergeler:
        """)
        
        tab1, tab2, tab3 = st.tabs(["Sunucu Deployment", "Cloud Deployment", "API Entegrasyonu"])
        
        with tab1:
            st.subheader("Sunucu Üzerinde Deployment")
            
            st.markdown("""
            ### Linux Sunucuda Deployment
            
            1. **Gerekli Kütüphaneleri Yükleyin**:
            ```bash
            pip install streamlit pandas numpy matplotlib seaborn scikit-learn PyPDF2 python-docx
            pip install aif360 fairlearn shap lime eli5
            ```
            
            2. **Uygulamayı Çalıştırın**:
            ```bash
            streamlit run app.py --server.port=8501 --server.address=0.0.0.0
            ```
            
            3. **Sürekli Çalışması İçin systemd Service Oluşturun**:
            ```bash
            sudo nano /etc/systemd/system/cvbias.service
            ```
            
            Service dosyası içeriği:
            ```
            [Unit]
            Description=CV Bias Analyzer
            After=network.target
            
            [Service]
            User=yourusername
            WorkingDirectory=/path/to/app
            ExecStart=/path/to/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
            Restart=always
            RestartSec=5
            
            [Install]
            WantedBy=multi-user.target
            ```
            
            Servisi etkinleştirme:
            ```bash
            sudo systemctl enable cvbias.service
            sudo systemctl start cvbias.service
            ```
            
            4. **NGINX ile Reverse Proxy Ayarlayın**:
            ```bash
            sudo apt install nginx
            sudo nano /etc/nginx/sites-available/cvbias
            ```
            
            NGINX config içeriği:
            ```
            server {
                listen 80;
                server_name your-domain.com;
                
                location / {
                    proxy_pass http://localhost:8501;
                    proxy_http_version 1.1;
                    proxy_set_header Upgrade $http_upgrade;
                    proxy_set_header Connection 'upgrade';
                    proxy_set_header Host $host;
                    proxy_cache_bypass $http_upgrade;
                }
            }
            ```
            
            Siteyi etkinleştirme:
            ```bash
            sudo ln -s /etc/nginx/sites-available/cvbias /etc/nginx/sites-enabled
            sudo systemctl restart nginx
            ```
            
            5. **SSL Sertifikası Ekleyin**:
            ```bash
            sudo apt install certbot python3-certbot-nginx
            sudo certbot --nginx -d your-domain.com
            ```
            """)
        
        with tab2:
            st.subheader("Cloud Deployment Seçenekleri")
            
            st.markdown("""
            ### 1. Streamlit Cloud
            
            En kolay deployment yöntemi:
            
            1. GitHub'a kodunuzu push edin
            2. [Streamlit Cloud](https://streamlit.io/cloud) hesabı oluşturun
            3. New app > GitHub repo > Main file: app.py
            4. Deploy app
            
            ### 2. Heroku
            
            1. Heroku hesabı oluşturun
            2. requirements.txt dosyası hazırlayın
            3. Procfile oluşturun: `web: streamlit run app.py --server.port=$PORT`
            4. Heroku CLI ile deploy edin:
            
            ```bash
            heroku login
            heroku create cv-bias-analyzer
            git push heroku main
            ```
            
            ### 3. AWS Elastic Beanstalk
            
            1. AWS hesabı oluşturun
            2. AWS CLI ve EB CLI yükleyin
            3. Uygulama dizininde:
            
            ```bash
            eb init -p python-3.8 cv-bias-analyzer
            eb create cv-bias-env
            eb deploy
            ```
            
            ### 4. Google Cloud Run
            
            1. Dockerfile oluşturun:
            
            ```Dockerfile
            FROM python:3.9-slim
            
            WORKDIR /app
            
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            
            COPY . .
            
            EXPOSE 8080
            
            CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
            ```
            
            2. GCP CLI ile deploy edin:
            
            ```bash
            gcloud builds submit --tag gcr.io/[PROJECT-ID]/cv-bias-analyzer
            gcloud run deploy --image gcr.io/[PROJECT-ID]/cv-bias-analyzer --platform managed
            ```
            """)
        
        with tab3:
            st.subheader("API Entegrasyonu")
            
            st.markdown("""
            ### CV Bias Analiz API'si Oluşturma
            
            Streamlit uygulaması yerine, sisteminizi bir API olarak da sunabilirsiniz:
            
            1. **FastAPI ile API Oluşturma**:
            
            ```python
            from fastapi import FastAPI, File, UploadFile
            import pandas as pd
            import io
            
            app = FastAPI(title="CV Bias Analyzer API")
            
            @app.post("/analyze-bias/")
            async def analyze_bias(file: UploadFile = File(...), 
                                 protected_attribute: str = "cinsiyet",
                                 target_col: str = "ise_alindi"):
                # CSV dosyasını oku
                content = await file.read()
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                
                # Bias analizi yap
                results = calculate_disparate_impact(df, protected_attribute, target_col)
                
                return {
                    "bias_detected": results['Bias Tespit Edildi'].any(),
                    "metrics": results.to_dict(orient="records")
                }
            
            @app.post("/mitigate-bias/")
            async def mitigate_bias(file: UploadFile = File(...),
                                  protected_attribute: str = "cinsiyet",
                                  target_col: str = "ise_alindi",
                                  method: str = "reweighing"):
                # Bias azaltma işlemleri
                # ...
                
                return {"status": "success", "method": method}
            ```
            
            2. **Docker ile API Deployment**:
            
            ```Dockerfile
            FROM python:3.9-slim
            
            WORKDIR /app
            
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            
            COPY . .
            
            EXPOSE 8000
            
            CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
            ```
            
            3. **API Gateway ve Lambda ile Serverless Deployment**:
            
            AWS Lambda ile API'yi serverless olarak sunabilirsiniz:
            
            ```python
            def lambda_handler(event, context):
                # API Gateway'den gelen request'i işle
                # Bias analizi yap
                # Sonuçları döndür
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'bias_detected': True,
                        'metrics': [...]
                    })
                }
            ```
            
            4. **Mevcut Sistemlere Entegrasyon**:
            
            Bias analiz API'nizi şu sistemlere entegre edebilirsiniz:
            
            - ATS (Applicant Tracking Systems)
            - İnsan Kaynakları Yönetim Sistemleri
            - Özgeçmiş Tarama Yazılımları
            - İş İlanı Platformları
            - Şirket İçi Talent Management Sistemleri
            """)
        
        st.subheader("VerifyWise Platformu Entegrasyonu")
        
        st.markdown("""
        ## VerifyWise ile Entegrasyon
        
        CV Bias Analiz aracını VerifyWise platformuna entegre etmek için önerilen adımlar:
        
        ### 1. Teknik Entegrasyon
        
        - **API Tabanlı Entegrasyon**: CV Bias analiz motorunu RESTful API olarak sunun
        - **SDK Geliştirme**: VerifyWise için özel Python/JavaScript SDK hazırlayın
        - **Webhook Desteği**: Gerçek zamanlı bias uyarıları için webhook mekanizması ekleyin
        
        ### 2. Veri Akışı
        
        - VerifyWise platformundan gelen CV'leri otomatik analiz edin
        - Bias sonuçlarını ve metrikleri VerifyWise dashboard'una gönderin
        - Periyodik bias raporlarını otomatik oluşturun
        
        ### 3. Özel VerifyWise Modülleri
        
        - **Bias Monitör Dashboard**: Zaman içindeki bias metriklerini izleme
        - **EU AI Act Uyumluluk Paneli**: Düzenleyici gereksinimleri takip etme
        - **Düzeltici Aksiyon Öneri Modülü**: Tespit edilen bias'ları azaltma önerileri
        
        ### 4. Ölçeklenebilirlik
        
        - Microservice mimarisi ile componentları ayırın
        - Kubernetes ile container orchestration sağlayın
        - Auto-scaling ile yüksek trafik dönemlerinde performansı koruyun
        
        ### Önerilen Zaman Çizelgesi
        
        | Aşama | Süre | Aktiviteler |
        |-------|------|-------------|
        | **PoC** | 4 Hafta | Temel bias analiz motorunun geliştirilmesi |
        | **MVP** | 8 Hafta | API entegrasyonu ve ilk VerifyWise modülü |
        | **Beta** | 12 Hafta | Tam EU AI Act uyumluluğu ve test sürümü |
        | **Lansman** | 16 Hafta | Production ortamına geçiş ve ilk müşterilere sunum |
        """)
        
        st.success("""
        VerifyWise platformuyla entegre edilmiş CV Bias Analiz aracı, şirketlerin EU AI Act uyumluluğunu sağlamalarına 
        yardımcı olurken, aynı zamanda daha adil işe alım süreçleri oluşturmalarını sağlayacaktır.
        """)
