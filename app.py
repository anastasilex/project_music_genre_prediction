from model import open_data, preprocess_data, split_data, load_model_and_predict
import streamlit as st 
import pandas as pd
import plotly.express as px
import seaborn as sns
import association_metrics as am


def process_main_page():
  show_main_page()
  
def page1():
    df = open_data()
    st.title('Обзор данных о музыкальных треках')
    st.write('Описания признаков, базовые статистики, корелляция')
    tab1, tab2, tab3, tab4 = st.tabs(["Общая информация о датасете", "Числовые признаки", "Категориальные признаки", "Анализ корелляций"])

    with tab1:
        st.header("Базовые статистики о датасете")
        row_count = df.shape[0]
        column_count = df.shape[1]
        duplicates = df[df.duplicated()]
        duplicate_row_count =  duplicates.shape[0]
        missing_value_row_count = df[df.isna().any(axis=1)].shape[0]
        table_markdown = f"""
            | Описание | Значение |
            |---|---|
            | Количество строк | {row_count} |
            | Количество столбцов | {column_count} |
            | Количество дубликатов | {duplicate_row_count} |
            | Количество строк с пропусками | {missing_value_row_count} |            
            """

        st.markdown(table_markdown)
        st.header("Описание признаков")

        columns = list(df.columns)
        df_for_columns_dtype = df.dropna()
        description_list = ['уникальный идентификатор трека',
                            'название трека',
                            'акустичность',
                            'танцевальность',
                            'продолжительность в милисекундах',
                            'энергичность',
                            'инструментальность',
                            'тональность',
                            'трек исполнен в записи или "вживую"',
                            'громкость',
                            'мажорная или минорная тональность',
                            'выразительность',
                            'темп',
                            'дата загрузки в сервис',
                            'привлекательность произведения для пользователей сервиса',
                            'музыкальный жанр'
                            ]

        column_info_table = pd.DataFrame({
            "Признак": columns,
            "Тип данных признака": df_for_columns_dtype.dtypes.tolist(),
            "Описание признака" : description_list
           })

        st.dataframe(column_info_table, hide_index=True)

    with tab2:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected_num_col = st.selectbox("Выберите признак", numeric_cols)
        st.header(f"Базовые статистики признака {selected_num_col}")
        col_info = {}
        col_info["Количество уникальных значений"] = len(df[selected_num_col].unique())
        col_info["Количество строк с пропусками"] = df[selected_num_col].isnull().sum()
        col_info["Среднее значение"] = df[selected_num_col].mean()
        col_info["Стандартное отклонение"] = df[selected_num_col].std()
        col_info["Минимальное значение"] = df[selected_num_col].min()
        col_info["Максимальное значение"] = df[selected_num_col].max()
        col_info["Медиана"] = df[selected_num_col].median()

        info_df = pd.DataFrame(list(col_info.items()), columns=['Описание', 'Значение'])

        st.dataframe(info_df)
        st.header(f"Гистограмма признака {selected_num_col}")
        fig = px.histogram(df, x=selected_num_col)
        st.plotly_chart(fig, use_container_width=True)

        st.header(f"""
                  Кривая плотности распределения признака {selected_num_col}"
                  по музыкальным жанрам
                  """)
        plot = sns.kdeplot(data=df, x=f'{selected_num_col}', hue='music_genre')
        st.pyplot(plot.get_figure())

    with tab3:
        cat_cols = df.select_dtypes(include='object')
        cat_cols_names = cat_cols.columns.tolist()

        selected_cat_col = st.selectbox("Выберите признак", cat_cols_names)
        st.header(f"{selected_cat_col}")
        cat_col_info = {}
        cat_col_info["Количество уникальных значений"] = len(df[selected_cat_col].unique())
        cat_col_info["Количество строк с пропусками"] = df[selected_cat_col].isnull().sum()
        cat_col_info["Наиболее часто встречающееся значение"] = df[selected_cat_col].mode()[0]

        cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Описание', 'Значение'])
        st.dataframe(cat_info_df)

        st.header(f"Гистограмма признака {selected_cat_col}")
        if selected_cat_col == 'track_name':
            popular_tracks = df['track_name'].value_counts().head(10).index
            fig = px.histogram(df[df['track_name'].isin(list(popular_tracks))], x=selected_cat_col)
            st.plotly_chart(fig, use_container_width=True)
        else:            
            fig = px.histogram(df, x=selected_cat_col)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Тепловая карта корелляций количественных признаков")
        st.write('Корелляция Пирсона показывает линейную корелляцию между признаками')
        fig = px.imshow(df[numeric_cols].corr(),  text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        df_cat = df.apply(lambda x: x.astype("category") if x.dtype == "object" else x)
        cramersv = am.CramersV(df_cat)
        v_measure = cramersv.fit()

        st.header("Матрица V-мер Крамера для категориальных признаков")
        st.write('V-мера Крамера позволяет оценить силу связи между двумя категориальными признаками')
        fig = px.imshow(v_measure,  text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        
def page2():
    process_side_bar_inputs()
    
def show_main_page():
    st.sidebar.title('Меню приложения')
    page = st.sidebar.radio('Переключить страницу', ['Обзор данных', 'Предсказание музыкального жанра'])

    if page == 'Обзор данных':
        page1()
    elif page == 'Предсказание музыкального жанра':
        page2()

def write_user_data(dict_input):
    st.write("## Ваши данные")
    input_markdown = f"""
            | Описание | Значение |
            |---|---|
            | Идентификатор трека | {dict_input.get('instance_id')} |
            | Акустичность | {dict_input.get('acousticness')} |
            | Танцевальность | {dict_input.get('danceability')} |
            | Длительность в мс | {dict_input.get('duration_ms')} |
            | Энергичность | {dict_input.get('energy')} |
            | Инструментальность | {dict_input.get('instrumentalness')} |
            | Вероятность 'живого' исполнения | {dict_input.get('liveness')} |
            | Громкость, дБ | {dict_input.get('loudness')} |
            | Интенсивность вокальной составляющей | {dict_input.get('speechiness')} |
            | Темп | {dict_input.get('tempo')} |
            | Привлекательность | {dict_input.get('valence')} |
            | Название трека | {dict_input.get('track_name')} |
            | Тональность | {dict_input.get('key')} |
            | Мажор/минор | {dict_input.get('mode')} |
            | Дата направления в сервис | {dict_input.get('obtained_date')} |           
            """
    st.markdown(input_markdown)
        
def write_prediction(prediction):
    st.write("## Предсказание")
    pred_markdown = f"""
            | Описание | Значение |
            |---|---|
            | Предсказанный музыкальный жанр | {prediction[0]} |           
            """
    st.markdown(pred_markdown)
    
def sidebar_input_features():
    count=1    
    instance_id = st.sidebar.slider(
      "Идентификатор трека",
      min_value=20000, max_value=100000, value=25000, step=1, key = count)
    count+=1
    acousticness = st.sidebar.slider(
      "Акустичность",
      min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    danceability = st.sidebar.slider(
      "Танцевальность",
      min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    duration_ms = st.sidebar.slider(
      "Длительность в мс",
      min_value=0, max_value=4500000, value=1000, step=1, key = count)
    count+=1
    energy = st.sidebar.slider(
      "Энергичность",
      min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    instrumentalness = st.sidebar.slider(
      "Инструментальность",
      min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    liveness = st.sidebar.slider(
        "Вероятность 'живого' исполнения",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    loudness = st.sidebar.slider(
        "Громкость, дБ",
        min_value=-45, max_value=0, value=-20, step=1, key = count)
    count+=1
    speechiness = st.sidebar.slider(
        "Интенсивность вокальной составляющей",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    tempo = st.sidebar.slider(
        "Темп",
        min_value=35, max_value=220, value=100, step=1, key = count)
    count+=1
    valence = st.sidebar.slider(
        "Привлекательность",
        min_value=0.0, max_value=1.0, value=0.5, step=0.001, key = count)
    count+=1
    track_name = st.sidebar.text_input("Название трека", key = count)
    count+=1
    key = st.sidebar.selectbox("Тональность",
                              ('D', 'A', 'E', 'G#', 'C', 'D#', 'A#', 'F', 'F#', 'G', 'C#', 'B'), key = count)
    count+=1
    mode = st.sidebar.selectbox("Мажорная или минорная тональность", ("Мажор", "Минор"), key = count)
    count+=1
    obtained_date = st.sidebar.selectbox("Дата направления в сервис", ("1 апреля", "3 апреля", '4 апреля', '5 апреля'), key = count)

    translatition = {
        "Мажор": "Major",
        "Минор": "Minor",
        "1 апреля": "1-Apr",
        "3 апреля": "3-Apr",
        "4 апреля": "4-Apr",
        "5 апреля": '5-Apr'
    }

    data = {
        "instance_id": instance_id,
        "acousticness": acousticness,
        "danceability": danceability,
        "duration_ms": duration_ms,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "loudness": loudness,
        "speechiness": speechiness,
        "tempo": tempo,
        "valence": valence,
        "track_name": track_name,
        "key": key,
        "mode": translatition[mode],
        "obtained_date": translatition[obtained_date]
    }

    df = pd.DataFrame(data, index=[0])
    return df, data

def process_side_bar_inputs():
    st.title('Предсказание музыкального жанра')
    with st.spinner('Подождите...'):
        st.sidebar.header('Заданные пользователем параметры')
        user_input_df, data_to_write = sidebar_input_features()
        train_df = open_data()
        train_X_df, _ = split_data(train_df)
        full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
        preprocessed_X_df = preprocess_data(full_X_df)

        user_X_df = preprocessed_X_df[:1]    
    
    
        prediction = load_model_and_predict(user_X_df)        
    
    
    write_prediction(prediction)
    write_user_data(data_to_write)
    
if __name__ == "__main__":
    process_main_page()                            