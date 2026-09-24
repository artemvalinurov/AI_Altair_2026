import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st

# ============================================================
# 1. АРХИТЕКТУРА ИМЕННО 4D CNN (Должна строго совпадать с обучением)
# ============================================================
class CustomGalaxy4DCNN(nn.Module):
    def __init__(self):
        super(CustomGalaxy4DCNN, self).__init__()
        # Блок 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Блок 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Блок 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Блок 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Глобальный средний пулинг
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Полносвязный классификатор строго под 4 бинарных выхода
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 4), # 4 выхода
            nn.Sigmoid()       # Обрезает выходы в [0..1]
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

# Кэшируем загрузку весов CNN
@st.cache_resource
def load_galaxy_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomGalaxy4DCNN()
    
    # Файл весов CNN должен лежать в этой же папке на ПК
    weights_path = "best_custom_4d_cnn.pth"
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device, weights_path
    else:
        return None, device, None

model, device, used_path = load_galaxy_model()

# Конвейер обработки (точно такой же, как val_tfms при обучении)
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 2. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ STREAMLIT
# ============================================================
st.set_page_config(page_title="Galaxy Classifier CNN", page_icon="🌌", layout="centered")

st.title("🌌 Локальный анализатор галактик (Кастомная CNN)")
st.write("Загрузите снимок космического объекта со своего компьютера для мгновенного анализа морфологии через Свёрточную Сеть.")

if model is not None:
    st.success(f"🤖 Свёрточная нейросеть (CNN) готова к работе. Файл весов: `{used_path}`")
else:
    st.error("❌ Файл весов `best_custom_4d_cnn.pth` не найден в текущей директории! Положите его рядом со скриптом.")

uploaded_file = st.file_uploader("Выберите файл изображения...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Разделение интерфейса на две удобные колонки
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Входное изображение", use_container_width=True)
        
    with col2:
        st.subheader("Морфологический профиль CNN:")
        
        # Инференс (предсказание)
        input_tensor = eval_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().squeeze(0).numpy()
        
        # 🌟 ЗДЕСЬ СТРОГО ИСПРАВЛЕНЫ ИНДЕКСЫ ДЛЯ ПРОГРЕСС-БАРОВ (0, 1, 2, 3)
        st.write(f"**Форма:** Спиральная ({pred[0]*100:.1f}%) vs Эллиптическая ({(1-pred[0])*100:.1f}%)")
        st.progress(float(pred[0]))
        
        st.write(f"**Ракурс диска:** С ребра ({pred[1]*100:.1f}%) vs Плашмя ({(1-pred[1])*100:.1f}%)")
        st.progress(float(pred[1])) # Теперь берёт строго индекс 1
        
        st.write(f"**Центральное ядро:** Есть бар ({pred[2]*100:.1f}%) vs Нет бара ({(1-pred[2])*100:.1f}%)")
        st.progress(float(pred[2])) # Теперь берёт строго индекс 2
        
        st.write(f"**Динамика:** Слияние галактик ({pred[3]*100:.1f}%) vs Стабильная ({(1-pred[3])*100:.1f}%)")
        st.progress(float(pred[3])) # Теперь берёт строго индекс 3
        
        # СЕКЦИЯ ТЕКСТОВОГО ВЕРДИКТА С АДАПТИВНЫМИ ПОРОГАМИ
        st.subheader("Текстовый вердикт ИИ:")
        
        # Снизили порог для формы до 0.3, чтобы сеть активнее реагировала на рукава спиралей
        форма = "спиральная" if pred[0] >= 0.3 else "эллиптическая"
        ракурс = "видна с ребра" if pred[1] >= 0.5 else "развернута плашмя"
        бар = "с перемычкой (баром)" if pred[2] >= 0.5 else "без бара"
        состояние = "в процессе слияния с сопоставимым соседом" if pred[3] >= 0.5 else "стабильная изолированная галактика"
        
        if форма == "эллиптическая":
            st.info(f"Объект классифицирован как **{форма}** галактика. Признаки структуры диска отсутствуют. Оценивается как **{состояние}**.")
        else:
            st.info(f"Объект классифицирован как **{форма}** галактика, которая **{ракурс}** и сформирована **{бар}**. Объект оценивается как **{состояние}**.")