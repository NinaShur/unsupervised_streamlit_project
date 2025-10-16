import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.title("Прогнозирование временного ряда с Prophet")

# загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV с двумя столбцами: ds (дата) и y (значение)", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # проверяем наличие нужных колонок
    if not {'ds', 'y'}.issubset(data.columns):
        st.error("Файл должен содержать колонки 'ds' (дата) и 'y' (значения)")
    else:
        data['ds'] = pd.to_datetime(data['ds'])
        st.write("Загруженные данные:")
        st.dataframe(data.head())

        # разделение на train/test по дате (последние 20% данных - тест)
        split_idx = int(len(data)*0.8)
        data_train = data.iloc[:split_idx]
        data_test = data.iloc[split_idx:]
        
        # обучение модели
        model = Prophet()
        model.fit(data_train)
        
        # создаем будущий датафрейм для теста + небольшой запас вперед
        future_periods = len(data_test) + 5  
        future = model.make_future_dataframe(periods=future_periods, freq='D')  # можно менять freq
        
        forecast = model.predict(future)
        
        # получаем прогноз на тестовой выборке
        forecast_test = forecast.iloc[split_idx:split_idx+len(data_test)]
        
        # метрики
        mae = mean_absolute_error(data_test['y'], forecast_test['yhat'])
        r2 = r2_score(data_test['y'], forecast_test['yhat'])
        
        st.write(f"MAE на тестовой выборке: {mae:.2f}")
        st.write(f"R² на тестовой выборке: {r2:.2f}")
        
        # график
        st.write("График прогноза и реальных данных")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['ds'], data['y'], label='Реальные данные', marker='o')
        ax.plot(forecast['ds'], forecast['yhat'], label='Прогноз', linestyle='--')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4, label='Интервал доверия')
        ax.legend()
        ax.set_xlabel("Дата")
        ax.set_ylabel("Значение")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # простая интерпретация
        st.write("""
        - Прогноз построен моделью Prophet, которая учитывает тренды и сезонность
        - MAE показывает среднюю ошибку прогноза
        - R² - насколько хорошо модель объясняет вариацию данных""")
else:
    st.info("Загрузите CSV файл с временным рядом для начала работы.")


        # - Область между `yhat_lower` и `yhat_upper` - интервал доверия прогноза.
        # 