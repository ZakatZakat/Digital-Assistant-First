from src.utils.check_serp_response import check_api_keys
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

# Заголовок приложения
st.title("Проверка статуса API-ключей")
st.write("Это приложение позволяет просматривать и обновлять статус ваших API-ключей.")

# Загрузка начального DataFrame
df = check_api_keys('api_keys_status.xlsx')

# Отображение текущей таблицы
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(min_column_width=100)
gb.configure_grid_options(domLayout='normal')
grid_options = gb.build()

AgGrid(
    df,
    gridOptions=grid_options,
    fit_columns_on_grid_load=True,
    columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
    height=450,
    width='100%',
    key="initial_grid"
)

# Раздел для добавления новых записей
st.subheader("Добавить новую запись")

# Поля ввода для новых данных
new_name = st.text_input("Имя")
new_email = st.text_input("Email")
new_api_key = st.text_input("API-ключ")
new_number = st.number_input("Номер", min_value=0)

# Кнопка для добавления новой записи
if st.button('Добавить новую запись'):
    if new_name and new_email and new_api_key:  # Проверка, что обязательные поля заполнены
        new_row = {
            'Имя': new_name,
            'Email': new_email,
            'API-ключ': new_api_key,
            'Номер': new_number
        }
        # Создание нового DataFrame для новой строки
        new_df = pd.DataFrame([new_row])
        
        # Объединение новой строки с существующим DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Сохранение обновленного DataFrame в файл Excel (или другое хранилище)
        df.to_excel('api_keys_status.xlsx', index=False)
        
        st.success("Новая запись успешно добавлена!")
        
        # Отображение обновленной таблицы
        AgGrid(
            df,
            gridOptions=grid_options,
            fit_columns_on_grid_load=True,
            columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
            height=450,
            width='100%',
            key="updated_grid"
        )
    else:
        st.error("Пожалуйста, заполните все обязательные поля (Имя, Email, API-ключ).")

# Кнопка для обновления статуса API-ключей
if st.button('Обновить статус API'):
    df = check_api_keys('api_keys_status.xlsx')
    
    AgGrid(
        df,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
        height=450,
        width='100%',
        key="updated_grid"
    )