import sys
from unittest.mock import MagicMock

# Патчинг torch.classes перед импортом Streamlit
sys.modules['torch._classes'] = MagicMock()
sys.modules['torch.classes'] = MagicMock()




import streamlit as st
import spacy_streamlit

import pandas as pd
import spacy





nlp = spacy.load('en_example_pipeline_ner_new')





# Заголовок с описанием
st.title("🏥 Анализатор медицинских текстов")
st.markdown("""
Анализ именованных сущностей (NER) в медицинских историях болезни и реабилитационных записях.
Поддерживает распознавание заболеваний, симптомов, лекарств и других медицинских понятий.
""")




      # Таблица результатов
       # Функция для извлечения сущностей
def extract_entities(text):
    doc = nlp(text)
    
    # Инициализируем словарь для хранения результатов
    entities = {
        "Scale": [],
        "Score": [],
        "Diagnos": [],
        "Psy": [],
        "Symptom": [],
        "Drug": []
    }
    
    # Сопоставление типов сущностей с колонками
    entity_mapping = {
        "MED_SCALES": "Scale",
        "SCORE": "Score",
        "DISEASE": "Diagnos",
        "PSY_TERMS": "Psy",
        "SYMPTOM": "Symptom",
        "DRUG": "Drug"
    }
    

    
    # Собираем сущности
    for ent in doc.ents:
        if ent.label_ in entity_mapping:
            column = entity_mapping[ent.label_]
            entities[column].append(ent.text)
    

    
    return entities



# Секция ввода текста
with st.expander("✍️ Ввод текста", expanded=True):
    input_method = st.radio(
        "Способ ввода:",
        ["Текстовое поле", "Загрузка файла"],
        horizontal=True
    )

    text = ""
    
    if input_method == "Текстовое поле":
        default_text = """Гипертоническая болезнь III стадии, 3 степени, риск ССО 4. ИБС, атеросклероз коронарных артерий, аорты. Шкала Ривермид (mRi): 13б. Шкала Рэнкина (mRs): 3б. ШРМ: 3б. FIM 112б."""
        
        text = st.text_area(
            "Введите медицинский текст:",
            value=default_text,
            height=200,
            help="Вставьте текст истории болезни или выписки"
        )
    else:
        uploaded_file = st.file_uploader(
            "Загрузите текстовый файл (.txt)",
            type=["txt"]
        )
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")

# Обработка текста
if st.button("🔍 Анализировать текст") and text:
    
    entities = extract_entities(text)
    with st.spinner("Анализ текста..."):
        

        

        doc = nlp(text)
        labales=['MED_SCALES', "SCORE", "DISEASE", "PSY_TERMS", "SYMPTOM", "DRUG"]
        colors = {

    "MED_SCALES": "#7457FF",
    "SCORE": "#008000",
    "DISEASE": "#F033FF" ,
     "PSY_TERMS": "#008B8B",
    "SYMPTOM": "#004B8B",
    "DRUG": "#2E8B57"
}



        spacy_streamlit.visualize_ner(
                     doc,
                    labels=labales,
    
                    show_table=False,  
                    title="Визуализатор реабилитационных сущностей NER",
                    colors=colors,
   
)

# создаем текст

    st.subheader("Все найденные сущности:")
    text_ent = " | ".join([f"{ent.text}" for ent in doc.ents])
    st.write(text_ent if text_ent else "Сущности не найдены")           
# Создаем таблицу
    max_len = max(len(v) for v in entities.values())
    table_data = []
    
    for i in range(max_len):
        row = {}
        for col in entities:
            row[col] = entities[col][i] if i < len(entities[col]) else ""
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Отображаем таблицу
    st.subheader("Результаты анализа")
    st.dataframe(
        df,
        column_config={
            "Scale": "Шкала",
            "Score": "Баллы",
            "Diagnos": "Диагноз",
            "Psy": "Псих. термин",
            "Symptom": "Симптом",
            "Drug": "Лекарство"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Скачивание результатов
    csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="Скачать как CSV",
        data=csv,
        file_name="medical_analysis.csv",
        mime="text/csv"
    )
