import os
import sys
import codecs
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st

# 🌟 МАГИЧЕСКАЯ СТРОКА ДЛЯ ИСПРАВЛЕНИЯ ПУТЕЙ НА ХОСТИНГЕ:
# Мы берем путь до папки, где лежит app.py, и принудительно добавляем её в поиск Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Теперь импорт сработает идеально как на Windows, так и на Linux-сервере Streamlit!
from model import CustomGalaxy4DCNN


# 1. МОБИЛЬНАЯ НАСТРОЙКА ЭКРАНА
st.set_page_config(
    page_title="Galaxy Classifier CNN",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. КЭШИРОВАННАЯ ЗАГРУЗКА НЕЙРОСЕТИ
@st.cache_resource
def load_galaxy_model():
    device = torch.device("cpu")
    model = CustomGalaxy4DCNN()
    weights_path = "best_custom_4d_cnn.pth"
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device, weights_path
    return None, device, None

model, device, used_path = load_galaxy_model()

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. ИНТЕРФЕЙС ВЕБ-САЙТА
st.title("🌌 Анализатор галактик (Кастомная CNN)")
st.write("Загрузите снимок космического объекта для мгновенного анализа морфологии.")

if model is not None:
    st.success("🤖 Нейросеть готова к анализу.")
else:
    st.error("❌ Файл весов `best_custom_4d_cnn.pth` не найден!")

uploaded_file = st.file_uploader("Выберите снимок галактики...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns()
    
    with col1:
        st.image(image, caption="Входной кадр", use_column_width=True)
        
    with col2:
        st.subheader("Морфологический профиль CNN:")
        
        input_tensor = eval_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).squeeze(0).numpy()
        
        st.write(f"**Форма:** Спиральная ({pred[0]*100:.1f}%) vs Эллиптическая ({(1-pred[0])*100:.1f}%)")
        st.progress(float(pred[0]))
        
        st.write(f"**Ракурс диска:** С ребра ({pred[1]*100:.1f}%) vs Плашмя ({(1-pred[1])*100:.1f}%)")
        st.progress(float(pred[1]))
        
        st.write(f"**Центральное ядро:** Есть бар ({pred[2]*100:.1f}%) vs Нет бара ({(1-pred[2])*100:.1f}%)")
        st.progress(float(pred[2]))
        
        st.write(f"**Динамика:** Слияние галактик ({pred[3]*100:.1f}%) vs Стабильная ({(1-pred[3])*100:.1f}%)")
        st.progress(float(pred[3]))
        
        st.subheader("Вердикт ИИ:")
        форма = "спиральная" if pred[0] >= 0.3 else "эллиптическая"
        ракурс = "видна с ребра" if pred[1] >= 0.5 else "развернута плашмя"
        бар = "с перемычкой (баром)" if pred[2] >= 0.5 else "без бара"
        состояние = "в процессе космического слияния" if pred[3] >= 0.5 else "стабильная изолированная галактика"
        
        if форма == "эллиптическая":
            st.info(f"Объект классифицирован как **{форма}** галактика. Внутренняя структура диска отсутствует. Оценивается как **{состояние}**.")
        else:
            st.info(f"Объект классифицирован как **{форма}** галактика, которая **{ракурс}** и сформирована **{бар}**. Объект оценивается как **{состояние}**.")

# ============================================================
# 4. АВТОМАТИЧЕСКОЕ ЧТЕНИЕ И СБОРКА ШАБЛОНА ИЗ 1.HTML И STYLE.CSS
# ============================================================
st.write("---")
st.subheader("🗺️ Справочник морфологии: Карта Эдвина Хаббла")

if os.path.exists("1.html") and os.path.exists("style.css"):
    # Открываем и читаем CSS и HTML файлы в правильной кодировке UTF-8
    with codecs.open("style.css", "r", "utf-8") as f:
        css_content = f.read()
    with codecs.open("1.html", "r", "utf-8") as f:
        html_content = f.read()
        
    # Соединяем их в единый блок стилизованной разметки
    full_component = f"<style>{css_content}</style>{html_content}"
    st.markdown(full_component, unsafe_allow_html=True)
else:
    st.warning("⚠️ Файлы `1.html` или `style.css` не найдены в корневой папке проекта!")
