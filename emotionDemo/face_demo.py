import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import torch
import numpy as np
from pathlib import Path
import face_alignment
from Model_architecture_Code import Block, ResNet

# --- GOOGLE STUN SUNUCULARI (Bağlantı Sorununu Çözen Kısım) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

EMOTION_LABELS = ["kızgın maş", "Iıııı", "Korkmuş faşat", "Mutlu Bun", "Kederli Ferot", "Çok İlginçmişş"]
WEIGHTS_PATH = Path("emotionDemo/Neconet_Weights3.pth")
IMAGE_SIZE = (64, 64)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = ResNet(Block, [2, 2, 2, 2], len(EMOTION_LABELS))
    state = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, device

@st.cache_resource
def load_fa():
    return face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")

model, device = load_model()
fa = load_fa()

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        scale = 0.4
        detect_frame = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        # Yüz bulma
        faces = fa.get_landmarks(detect_frame)

        if faces:
            landmarks = faces[0]
            min_x, min_y = np.min(landmarks[:, 0]) / scale, np.min(landmarks[:, 1]) / scale
            max_x, max_y = np.max(landmarks[:, 0]) / scale, np.max(landmarks[:, 1]) / scale
            
            x1, y1, x2, y2 = int(min_x), int(min_y), int(max_x), int(max_y)
            crop = img[max(0,y1):y2, max(0,x1):x2]
            
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, IMAGE_SIZE)
                tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
                tensor = (tensor - 0.5) / 0.5
                
                with torch.no_grad():
                    logits = model(tensor.to(device))
                    probs = torch.softmax(logits, dim=1).squeeze().numpy()
                    label = EMOTION_LABELS[np.argmax(probs)]
                
                # Kareyi ve etiketi çiz
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

st.title("Faraş Duygu Analizi")
st.write("Start'a bas ve kameraya bak..")

webrtc_streamer(
    key="emotion-live",
    mode=None, # VideoProcessor modunu otomatik seçer
    rtc_configuration=RTC_CONFIGURATION, # Hata veren STUN ayarı buraya eklendi
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}, # Sadece görüntü
    async_processing=True,
)

        
