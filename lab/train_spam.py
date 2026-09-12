"""
Детектор спама, русский + английский: SMS Spam Collection (UCI, ~5,2 тыс. уникальных сообщений)
в оригинале и в переводе на русский + фразы из частей (письменный спам, короткие русские сообщения).
Запуск из папки lab: python train_spam.py → spam_model.pth (скопировать в ../models).
"""
import csv
import random
import sys

import torch
from huggingface_hub import hf_hub_download

from text_classifier import DEVICE, accuracy, encode, train

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
torch.manual_seed(42)

LABELS = ["HAM", "SPAM"]

# В SMS Spam Collection нет русского и почти нет «письменного» спама — добавляем собранные из частей фразы
SPAM_PARTS = {
    "en": (["Congratulations!", "URGENT!", "Dear customer,", "Last chance!", "Exclusive offer:", "WINNER!"],
           ["You won a prize of $1000", "Earn money fast from home", "Get a free iPhone", "Your loan is approved",
            "Lose 10 kg in a week", "Claim your cash reward", "Cheap pills without prescription",
            "Your account will be blocked"],
           ["Click here now!", "Call 0800 123 456 to claim.", "Reply YES to join.", "No experience needed!",
            "Limited time only, act today!", "Visit www.win-prize.biz"]),
    "ru": (["Поздравляем!", "СРОЧНО!", "Уважаемый клиент,", "Только сегодня!", "Акция!", "Вы выиграли!"],
           ["Вы выиграли iPhone", "Заработок от 5000 рублей в день без вложений", "Кредит одобрен без справок",
            "Ваша карта заблокирована", "Похудей на 10 кг за неделю", "Получите приз 100 000 рублей",
            "Бесплатная консультация и скидка 90%"],
           ["Перейдите по ссылке!", "Звоните 8-800-555-35-35!", "Ответьте ДА, чтобы получить.",
            "Опыт не нужен!", "Предложение ограничено!", "Подробности на сайте win-prize.ru"]),
}
HAM = {
    "en": ["Hi, can we schedule a meeting tomorrow at 3pm?", "Please find the attached report.",
           "I'll be late, stuck in traffic.", "Don't forget to buy milk on the way home.",
           "The project deadline moved to Friday.", "Thanks for yesterday, it was great!",
           "Can you send me the slides from the lecture?", "Happy birthday! Have a wonderful day.",
           "Where are we meeting for lunch?", "Let me know if you have any questions."],
    "ru": ["Привет, завтра встречаемся в три?", "Отправляю отчёт во вложении.", "Задержусь, стою в пробке.",
           "Купи молоко по дороге домой.", "Дедлайн по проекту перенесли на пятницу.",
           "Спасибо за вчерашний вечер!", "Скинь, пожалуйста, презентацию с лекции.",
           "С днём рождения! Всего самого лучшего.", "Где встречаемся на обед?",
           "Мама, я дома, всё хорошо.", "Если будут вопросы — пиши.", "Во сколько завтра тренировка?"],
}


def synthetic(rng, n):
    out = []
    for _ in range(n):
        lang = rng.choice(["en", "ru"])
        a, b, c = SPAM_PARTS[lang]
        out.append((f"{rng.choice(a)} {rng.choice(b)}. {rng.choice(c)}", 1))
        out.append((" ".join(rng.sample(HAM[lang], rng.randint(1, 2))), 0))
    return out


def load_bilingual():
    """SMS Spam Collection в оригинале + машинный перевод на русский. Возвращает [(en, ru, label)] без дублей."""
    path = hf_hub_download("dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset", "data-augmented.csv",
                           repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen, out = set(), []
    for r in rows:
        en = r["text"].strip()
        if en and en not in seen and r["labels"] in ("ham", "spam"):
            seen.add(en)
            out.append((en, r["text_ru"].strip(), int(r["labels"] == "spam")))
    return out


if __name__ == "__main__":
    msgs = load_bilingual()
    rng = random.Random(42)
    rng.shuffle(msgs)
    n = len(msgs)
    # Делим по исходному сообщению, чтобы перевод теста не попал в обучение
    both = lambda part: [(en, y) for en, _, y in part] + [(ru, y) for _, ru, y in part if ru]
    train_m, val_m, test_m = msgs[:int(n * .8)], msgs[int(n * .8):int(n * .9)], msgs[int(n * .9):]
    train_p, val_p = both(train_m) + synthetic(rng, 600), both(val_m)
    rng.shuffle(train_p)
    xs = lambda p: [t for t, _ in p]
    ys = lambda p: [y for _, y in p]
    # Спама в 6 раз меньше, чем обычных сообщений
    weight = torch.tensor([1.0, 3.0])
    model, vocab = train(xs(train_p), ys(train_p), xs(val_p), ys(val_p), LABELS, "spam_model.pth",
                         max_len=64, vocab_size=30000, epochs=20, patience=4, class_weight=weight)
    for lang, test_p in (("EN", [(en, y) for en, _, y in test_m]), ("RU", [(ru, y) for _, ru, y in test_m if ru])):
        acc, pred = accuracy(model, encode(xs(test_p), vocab, 64), torch.tensor(ys(test_p)))
        y = torch.tensor(ys(test_p))
        tp = ((pred == 1) & (y == 1)).sum().item()
        print(f"\nТест {lang}: точность {acc:.3f} | найдено спама {tp}/{(y == 1).sum().item()} | "
              f"ложных срабатываний {((pred == 1) & (y == 0)).sum().item()} из {(y == 0).sum().item()}")
    for text in ["Congratulations! You won a prize of 1000000 dollars. Click here to claim your reward now!",
                 "Hi, can we schedule a meeting tomorrow at 3pm to discuss the quarterly report?",
                 "Earn money fast from home! No experience needed. Join our exclusive program today free!",
                 "Please find the attached document and let me know if you have any questions or feedback.",
                 "Вы стали победителем розыгрыша! Заберите свой приз по ссылке",
                 "Привет, я забыл ключи, откроешь дверь вечером?",
                 "Займы онлайн за 5 минут, без отказа, звоните прямо сейчас"]:
        with torch.no_grad():
            p = torch.softmax(model.eval()(encode([text], vocab, 64).to(DEVICE)), -1)[0]
        print(f"  {LABELS[int(p.argmax())]} {p[1]:.0%}  {text[:60]}")
