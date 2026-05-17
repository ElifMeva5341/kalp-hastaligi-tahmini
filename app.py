import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Sayfa yapılandırması
st.set_page_config(page_title="Kalp Hastalığı Risk Analizi", page_icon="❤️", layout="centered")

# Eğitilmiş modeli yükleme
@st.cache_resource
def load_model():
    with open("model_rf.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("Model dosyası yüklenemedi. Lütfen GitHub deponuzda 'model_rf.pkl' dosyasının olduğundan emin olun.")

# Başlık ve Açıklama
st.title("❤️ Kalp Hastalığı Teşhis ve Risk Analiz Paneli")
st.write("BIM 322 Makine Öğrenmesi ve Uygulamaları - Proje Ödevi")
st.markdown("---")

st.subheader("📋 Hasta Sağlık Parametrelerini Giriniz")
st.write("Lütfen hastaya ait en güçlü 4 klinik özelliği giriniz:")

# Kullanıcı Girdileri (Ortak özellikler: thal, thalach, cp, ca)
cp = st.selectbox(
    "Göğüs Ağrısı Tipi (cp)", 
    options=[1, 2, 3, 4],
    format_func=lambda x: {1: "1: Tipik Anjin", 2: "2: Atipik Anjin", 3: "3: Anjin Olmayan", 4: "4: Semptomsuz"}[x]
)

thalach = st.slider("Ulaşılan Maksimum Kalp Atış Hızı (thalach)", min_value=60, max_value=220, value=150)

ca = st.selectbox("Renklendirilen Ana Damar Sayısı (ca)", options=[0, 1, 2, 3])

thal = st.selectbox(
    "Talyum Sintigrafisi Sonucu (thal)", 
    options=[3.0, 6.0, 7.0],
    format_func=lambda x: {3.0: "3.0: Normal", 6.0: "6.0: Sabit Kusur", 7.0: "7.0: Geri Dönüşümlü Kusur"}[x]
)

# Tahmin Butonu
if st.button("🩺 Risk Durumunu Tahmin Et"):
    # Girdileri modele uygun formata getirme (Sıralama: thal, thalach, cp, ca)
    input_data = pd.DataFrame([[thal, thalach, cp, ca]], columns=['thal', 'thalach', 'cp', 'ca'])
    
    # Model tahmini
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Analiz Sonucu")
    
    if prediction[0] == 1:
        st.error(f"🚨 DİKKAT: Yüksek Kalp Hastalığı Riski Tespit Edildi! (Risk Oranı: %{prediction_proba*100:.1f})")
        st.write("**Öneri:** Hastanın en kısa sürede uzman bir kardiyoloğa yönlendirilmesi ve ileri tetkiklerin yapılması hayati önem taşımaktadır.")
    else:
        st.success(f"✅ TEBRİKLER: Düşük Kalp Hastalığı Riski. (Risk Oranı: %{prediction_proba*100:.1f})")
        st.write("**Öneri:** Sağlık parametreleri mevcut model sınırlarına göre normal görünmektedir. Rutin kontrollere devam edilmesi önerilir.")
