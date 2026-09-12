"""
Тональность отзывов (негатив / нейтрально / позитив) на датасете RuReviews:
45 тыс. отзывов о товарах для обучения, 15 тыс. для теста.
Запуск из папки lab: python train_sentiment.py → sentiment_model.pth (скопировать в ../models).
"""
import sys

import torch

from hf_data import load_parquet
from text_classifier import DEVICE, accuracy, encode, train

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
torch.manual_seed(42)

LABELS = ["негативный", "нейтральный", "позитивный"]
L2I = {"negative": 0, "neutral": 1, "positive": 2}
REPO, REV = "ai-forever/ru-reviews-classification", "refs/convert/parquet"


def split(name):
    rows = load_parquet(REPO, f"default/{name}/0000.parquet", REV)
    return [r["text"] for r in rows], [L2I[r["label_text"]] for r in rows]


if __name__ == "__main__":
    tr_x, tr_y = split("train")
    val_x, val_y = split("validation")
    test_x, test_y = split("test")
    model, vocab = train(tr_x, tr_y, val_x, val_y, LABELS, "sentiment_model.pth", max_len=96)
    acc, pred = accuracy(model, encode(test_x, vocab, 96), torch.tensor(test_y))
    print(f"\nТочность на тесте ({len(test_x)} отзывов): {acc:.3f}")
    for text in ["Отличный товар, очень доволен!", "Ужасное качество, полный брак.", "Нормально, ничего особенного.",
                 "Пришло быстро, но размер маломерит", "Не рекомендую, деньги на ветер"]:
        with torch.no_grad():
            p = torch.softmax(model.eval()(encode([text], vocab, 96).to(DEVICE)), -1)[0]
        print(f"  {text!r} → {LABELS[int(p.argmax())]} ({p.max():.0%})")
