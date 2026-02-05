import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
import face_alignment
from Model_architecture_Code import Block, ResNet
from PIL import Image

# --- AYARLAR ---
EMOTION_LABELS = ["kızgın maş", "Iıııı", "Korkmuş faşat", "Mutlu Bun", "Kederli Ferot", "Çok İlginçmişş"]
WEIGHTS_PATH = Path("emotionDemo/Neconet_Weights3.pth")
IMAGE_SIZE = (64, 64)

@st.cache_resource
def load_everything():
    device = torch.device("cpu")
    model = ResNet(Block, [2, 2, 2, 2], len(EMOTION_LABELS))
    # Dosya yolu kontrolü
    if not WEIGHTS_PATH.exists():
        st.error(f"Model ağırlıkları bulunamadı: {WEIGHTS_PATH}")
        return None, None, None
    state = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")
    return model, fa, device

st.set_page_config(page_title="Faraş Duygu Analizi", layout="centered")
st.title(" Faraş Duygu Analizi ")
st.write("Selfie Çek..")

model, fa, device = load_everything()

if model:
    # BU KISIM SIHİRLİ: Hem kamerayı açar hem de hata vermez.
    img_file = st.camera_input("Kamerayı Aç ve Çek")

    if img_file:
        # Görüntüyü işle
        image = Image.open(img_file)
        img = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        with st.spinner('Analiz ediliyor...'):
            scale = 0.4
            detect_frame = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)
            faces = fa.get_landmarks(detect_frame)

            if faces:
                landmarks = faces[0]
                min_x, min_y = np.min(landmarks[:, 0]) / scale, np.min(landmarks[:, 1]) / scale
                max_x, max_y = np.max(landmarks[:, 0]) / scale, np.max(landmarks[:, 1]) / scale
                
                x1, y1, x2, y2 = int(min_x), int(min_y), int(max_x), int(max_y)
                crop = img_bgr[max(0,y1):y2, max(0,x1):x2]
                
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, IMAGE_SIZE)
                    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
                    tensor = (tensor - 0.5) / 0.5
                    
                    with torch.no_grad():
                        logits = model(tensor.to(device))
                        probs = torch.softmax(logits, dim=1).squeeze().numpy()
                        label = EMOTION_LABELS[np.argmax(probs)]
                        conf = np.max(probs) * 100
                    
                    # Ekrana Şık Bir Sonuç Bas
                    st.success(f"Tahmin: **{label}** (Eminlik: %{conf:.1f})")
                    # Yüzün etrafına kutu çizip gösterelim
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Analiz Edilen Kare")
            else:
                st.warning("Yüz algılanamadı. Biraz daha yakından dener misin?")
