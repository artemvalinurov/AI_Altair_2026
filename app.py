import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st

# ============================================================
# 1. МОБИЛЬНАЯ НАСТРОЙКА ЭКРАНА (СТРОГО НА ПЕРВОЙ СТРОКЕ)
# ============================================================
st.set_page_config(
    page_title="Galaxy Classifier CNN",
    page_icon="🌌",
    layout="wide",                      # Растягивает интерфейс на весь экран телефона
    initial_sidebar_state="collapsed"   # Прячет боковое меню на смартфонах
)

# ============================================================
# 2. АРХИТЕКТУРА ИМЕННО 4D CNN (Должна строго совпадать с обучением)
# ============================================================
class CustomGalaxy4DCNN(nn.Module):
    def __init__(self):
        super(CustomGalaxy4DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 4), # 4 выхода под бинарные пары
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Кэшируем загрузку весов CNN, чтобы сервер не зависал
@st.cache_resource
def load_galaxy_model():
    # На бесплатном хостинге Streamlit Cloud нет GPU, поэтому ВСЕГДА принудительно ставим cpu
    device = torch.device("cpu")
    model = CustomGalaxy4DCNN()
    
    weights_path = "best_custom_4d_cnn.pth" # Имя вашего файла весов на GitHub
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device, weights_path
    else:
        return None, device, None

model, device, used_path = load_galaxy_model()

# Конвейер обработки картинок (точно такой же, как при обучении)
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# 3. АДАПТИВНЫЙ ИНТЕРФЕЙС STREAMLIT СМАРТФОН / ПК
# ============================================================
st.title("🌌 Анализатор галактик (Кастомная CNN)")
st.write("Загрузите снимок космического объекта для мгновенного анализа морфологии.")

if model is not None:
    st.success(f"🤖 Нейросеть готова к анализу.")
else:
    st.error("❌ Файл весов `best_custom_4d_cnn.pth` не найден в репозитории GitHub! Положите его рядом с app.py.")

# На телефоне эта кнопка автоматически предложит открыть Галерею или Камеру
uploaded_file = st.file_uploader("Выберите снимок галактики...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 🌟 На ПК это будут две колонки рядом. На телефоне они автоматически встанут друг под друга!
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # use_column_width=True заставляет картинку идеально сжиматься под ширину экрана смартфона
        st.image(image, caption="Входной кадр", use_column_width=True)
        
    with col2:
        st.subheader("Морфологический профиль CNN:")
        
        # Инференс нейросети
        input_tensor = eval_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).squeeze(0).numpy()
        
        # 🌟 ВСЕ ИНДЕКСЫ ИСПРАВЛЕНЫ И СЧИТАЮТСЯ НЕЗАВИСИМО
        st.write(f"**Форма:** Спиральная ({pred[0]*100:.1f}%) vs Эллиптическая ({(1-pred[0])*100:.1f}%)")
        st.progress(float(pred[0]))
        
        st.write(f"**Ракурс диска:** С ребра ({pred[1]*100:.1f}%) vs Плашмя ({(1-pred[1])*100:.1f}%)")
        st.progress(float(pred[1])) 
        
        st.write(f"**Центральное ядро:** Есть бар ({pred[2]*100:.1f}%) vs Нет бара ({(1-pred[2])*100:.1f}%)")
        st.progress(float(pred[2])) 
        
        st.write(f"**Динамика:** Слияние галактик ({pred[3]*100:.1f}%) vs Стабильная ({(1-pred[3])*100:.1f}%)")
        st.progress(float(pred[3])) 
        
        # ТЕКСТОВЫЙ ВЕРДИКТ С АДАПТИВНЫМ ПОРОГОМ ДЛЯ СПИРАЛЕЙ (0.3)
        st.subheader("Вердикт ИИ:")
        
        форма = "спиральная" if pred[0] >= 0.3 else "эллиптическая"
        ракурс = "видна с ребра" if pred[1] >= 0.5 else "развернута плашмя"
        бар = "с перемычкой (баром)" if pred[2] >= 0.5 else "без бара"
        состояние = "в процессе космического слияния" if pred[3] >= 0.5 else "стабильная изолированная галактика"
        
        if форма == "эллиптическая":
            st.info(f"Объект классифицирован как **{форма}** галактика. Внутренняя структура диска отсутствует. Оценивается как **{состояние}**.")
        else:
            st.info(f"Объект классифицирован как **{форма}** галактика, которая **{ракурс}** и сформирована **{bar}**. Объект оценивается как **{состояние}**.")
