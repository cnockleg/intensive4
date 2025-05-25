import pandas as pd
import streamlit as st
from project.model import Model

@st.cache_resource
def load_model():
    return Model(r'intensive4_evgeniy\final_model.pth')

model = load_model()

st.title("Классификация отзывов по тегам")
text_input = st.text_area("Введите отызв: ")

if st.button("Предсказать теги", type='primary'):
    if text_input.strip():
        tags, scores = model.predict(text_input)

        df_scores = pd.DataFrame({
            'Тег': model.labels,
            'Вероятность': [f"{s:.2f}" for s in scores]
        })

        st.subheader("Предсказанные теги: ")
        for tag in tags:
            st.markdown(f"- **{tag}**")
        
        st.subheader("Вероятность по тегам: ")
        # st.table(df_scores)
        st.dataframe(
            df_scores.style.background_gradient(cmap='gist_grey', subset=['Вероятность'])
        )
    else:
        st.warning("Сначала введите текст")