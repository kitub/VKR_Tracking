import streamlit as st
import cv2
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import altair as alt
from ultralytics import YOLO
import supervision as sv
import time

demo = True

# --- Настройка базы данных ---
conn = sqlite3.connect('people_count.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        timestamp TEXT
    )
''')
conn.commit()

# --- Генерация синтетических данных за последние 10 дней ---
# для демонстрации
if demo:
    c.execute("SELECT COUNT(*) FROM counts")
    if c.fetchone()[0] == 0:
        now = datetime.now()
        start = now - timedelta(days=10)
        total_minutes = int((now - start).total_seconds() // 60)
        # Среднее и отклонение для распределения количества пешеходов в минуту
        mean_per_min = 5
        std_per_min = 2
        for i in range(total_minutes):
            minute_time = start + timedelta(minutes=i)
            # Случайное количество появлений пешеходов в минуту
            count = max(0, int(np.random.normal(loc=mean_per_min, scale=std_per_min)))
            for _ in range(count):
                # Случайная секунда внутри минуты
                rand_sec = np.random.randint(0, 60)
                ts = (minute_time + timedelta(seconds=rand_sec)).isoformat()
                # Используем person_id = NULL, SQLite примет AUTOINCREMENT ID
                c.execute("INSERT INTO counts(person_id, timestamp) VALUES (?, ?)", (None, ts))
        conn.commit()

# --- Элементы боковой панели ---
st.title("Система трекинга пешеходов")

# 1) Выбор камеры с именами устройств
def list_cameras(max_devices=5):
    import sys
    devices = []
    # Windows
    if sys.platform.startswith('win'):
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            for idx, name in enumerate(graph.get_input_devices()):
                devices.append((name, idx))
        except Exception:
            for i in range(max_devices):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    devices.append((f"Камера {i}", i))
                    cap.release()
    # другие платформы
    else:
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append((f"Камера {i}", i))
                cap.release()
    return devices

# Выпадающий список: отображается имя, возвращается индекс
device_option = st.sidebar.selectbox(
    "Выберите камеру", list_cameras(),
    format_func=lambda x: x[0]
)
camera_index = device_option[1]

time_options = {
    '10 минут': 10,
    '30 минут': 30,
    '1 час': 60,
    '4 часа': 240,
    '1 день': 1440
}
interval_label = st.sidebar.selectbox("Интервал для гистограммы", list(time_options.keys()))
interval_minutes = time_options[interval_label]
    
# 2) Загрузка модели YOLO и перевод на CUDA
def load_model():
    model = YOLO('yolov8m.pt')
    model.to('cuda')
    return model

# Кеширование ресурса модели
load_model = st.cache_resource(load_model)
model = load_model()

# Инициализация трекера ByteTrack через supervision с пользовательскими гиперпараметрами
tracker = sv.ByteTrack(
    track_activation_threshold=0.2,
    minimum_matching_threshold=0.85,
    lost_track_buffer=67,
    minimum_consecutive_frames=1
)
box_annotator = sv.BoxAnnotator()
trace_annotator = sv.TraceAnnotator(thickness=3, trace_length=50)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.25, text_color=sv.Color.WHITE)

# Плейсхолдеры для видеопотока и гистограммы
if 'vid_pl' not in st.session_state:
    st.session_state.vid_pl = st.empty()

if 'hist_pl' not in st.session_state:
    st.session_state.hist_pl = st.empty()

st.session_state.hist_pl.empty()
st.session_state.vid_pl.empty()

# ограничения FPS
desired_fps = 30
frame_interval = 1.0 / desired_fps

# параметры обновления гистограммы
update_interval = 30  # секунд
last_hist_update = time.time() - update_interval

# 4) Обработка видео
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    st.error(f"Не удалось открыть камеру {camera_index}")
else:
    while True:            
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # Выполнение детекции
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Отфильтровать только людей (class_id==0)
        people = detections[detections.class_id == 0]
        # Трекинг
        detections = tracker.update_with_detections(detections=people)
        tracks = tracker.tracked_tracks
        # 3) Подсчет новых человек
        now = datetime.now()
        for tr in tracks:
            if tr.state == 0:
                c.execute(
                    "INSERT INTO counts(person_id, timestamp) VALUES (?, ?)",
                    (tr.id, now.isoformat())
                )
                conn.commit()
       # Формирование меток
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        # Аннотирование трейсами, рамками и метками
        annotated = frame.copy()
        annotated = trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        # Вывод
        st.session_state.vid_pl.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)


        # обновление гистограммы
        if time.time() - last_hist_update >= update_interval:
            st.session_state.hist_pl.empty()
            start = now - timedelta(minutes=interval_minutes*20)
            c.execute("SELECT timestamp FROM counts WHERE timestamp>=?", (start.isoformat(),))
            rows = c.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=['ts'])
                df['ts'] = pd.to_datetime(df['ts'])
                edges = pd.date_range(start, now, periods=21)
                counts = np.histogram(df['ts'].astype('int64')/1e9, bins=edges.astype('int64')/1e9)[0]
                hist_df = pd.DataFrame({
                    'time_start': edges[:-1],
                    'time_end': edges[1:],
                    'count': counts,
                    'zeros': np.zeros(counts.shape)
                })
                chart = alt.Chart(hist_df).mark_bar().encode(
                    x=alt.X('time_start:T', title='Время'),
                    x2='time_end:T',
                    y=alt.Y('count:Q', title='Количество', scale=alt.Scale(domainMin=0)),
                    y2='zeros:Q'

                ).properties(width=700, height=300)
                st.session_state.hist_pl.altair_chart(chart, use_container_width=True)
            else:
                st.session_state.hist_pl.write("Нет данных за выбранный период.")
            last_hist_update = time.time()

        # ограничение FPS
        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)