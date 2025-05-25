import torch
from sentence_transformers import SentenceTransformer

from project.modeling import MultiLabelClassifier

class Model():
    def __init__(self, model_path):
        self.model = MultiLabelClassifier(embedding_dim=312, num_labels=6)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.sbert = SentenceTransformer('cointegrated/rubert-tiny2')

        self.labels = [
            'Вопрос решен',
            'Нравится качество выполнения заявки',
            'Нравится качество работы сотрудников',
            'Нравится скорость отработки заявок',
            'Понравилось выполнение заявки',
            'Вопрос не решен'
        ]

        self.threshold = [0.31 for label in self.labels]

    def predict(self, text: str):
        # self.model.eval()
        with torch.no_grad():
            emb = self.sbert.encode(text, convert_to_tensor=True).unsqueeze(0)  
            preds = self.model(emb).cpu().numpy()[0]
            preds_bin = (preds >= self.threshold).astype(int)
            predicted_tags = [label for label, pred in zip(self.labels, preds_bin) if pred == 1]
        return predicted_tags, preds
    
if __name__ == "__main__":
    model = Model(r'intensiv-3-main\intensive4\intensive4_evgeniy\final_model.pth')
    text = 'оперативно приехали, все хорошо'

    tags, scores = model.predict(text)
    
    print(f"Текст: {text}")
    print(f"Предсказанные теги: {tags}")
    print(f"Raw scores: {scores}")