import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    return pd.read_excel(path)


def preprocess_data(data):
    tags = ['Вопрос решен', 'Вопрос не решен', 'Нравится качество выполнения заявки', 'Нравится качество работы сотрудников', 'Нравится скорость отработки заявок', 'Понравилось выполнение заявки']

    for tag in tags:
        data[tag] = data['taxonomy'].apply(lambda x: 1 if pd.notna(x) and tag in x else 0)

    delete_columns = ['annotation_id', 'annotator', 'id', 'created_at', 'taxonomy', 'lead_time', 'updated_at']
    data.drop(columns=delete_columns, inplace=True)

    data = data.drop(1194).reset_index(drop=True)
    data = data.drop(1308).reset_index(drop=True)
    data = data.drop(2098).reset_index(drop=True)

    data['comment_clean'] = data['comment'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    data = data.drop_duplicates(subset=['comment_clean'])
    return data


def plot_class_distribution(data_clean):
    class_columns = data_clean.columns[2:-1]  
    class_distribution = data_clean[class_columns].sum()

    plt.figure(figsize=(10, 5))
    class_distribution.plot(kind="bar", rot=70)
    plt.title("Распределение классов")
    plt.ylabel("Количество")
    plt.show()