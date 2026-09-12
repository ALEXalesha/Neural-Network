"""
Сравнение качества переводчика в обе стороны на одних и тех же 3000 парах Tatoeba,
которые модель не видела при обучении (та же отложенная выборка, что в train_translator_bpe.py).

Запуск из папки lab: python eval_translator.py [папка с моделями, по умолчанию ../models]
BLEU и chrF (sacrebleu): чем больше, тем лучше. Loss при обучении так сравнивать нельзя —
русский с окончаниями генерировать труднее, поэтому у EN→RU он всегда выше.
"""
import sys
import time
from pathlib import Path

import sacrebleu
import torch
from tokenizers import Tokenizer

from train_translator_bpe import DEVICE, TranslatorBPE, build_pairs, translate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

N_VAL = 3000


def load(models, direction, src_tok, tgt_tok):
    ck = torch.load(models / f"translator_bpe_{direction}.pth", weights_only=True, map_location=DEVICE)
    model = TranslatorBPE(src_tok.get_vocab_size(), tgt_tok.get_vocab_size(), **ck["config"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    return model.eval()


def main():
    models = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "models"
    en_tok = Tokenizer.from_file(str(models / "bpe_en.json"))
    ru_tok = Tokenizer.from_file(str(models / "bpe_ru.json"))
    pairs = build_pairs()
    val, train = pairs[:N_VAL], pairs[N_VAL:]
    # В Tatoeba у одной фразы бывает несколько переводов: 55% английских фраз отложенной выборки
    # встречаются в обучении с другим переводом. Честное сравнение — на парах, где обе стороны новые.
    seen_en, seen_ru = {e for e, _ in train}, {r for _, r in train}
    clean = [i for i, (e, r) in enumerate(val) if e not in seen_en and r not in seen_ru]
    print(f"Отложенных пар: {len(val)}, из них полностью новых для модели: {len(clean)}")
    en, ru = [p[0] for p in val], [p[1] for p in val]
    for direction, src, ref, st, tt in (("en2ru", en, ru, en_tok, ru_tok), ("ru2en", ru, en, ru_tok, en_tok)):
        model = load(models, direction, st, tt)
        t0 = time.time()
        hyp = [translate(model, s, st, tt) for s in src]
        for name, idx in (("все пары", range(len(val))), ("только новые", clean)):
            h, r = [hyp[i] for i in idx], [ref[i] for i in idx]
            bleu = sacrebleu.corpus_bleu(h, [r]).score
            chrf = sacrebleu.corpus_chrf(h, [r]).score
            exact = sum(a.strip() == b.strip() for a, b in zip(h, r)) / len(r)
            print(f"{direction} [{name}]: BLEU {bleu:.1f} | chrF {chrf:.1f} | дословно совпало {exact:.0%}")
        print(f"    {time.time() - t0:.0f} с")
        for i in clean[:3]:
            print(f"    {src[i]} → {hyp[i]}   (эталон: {ref[i]})")


if __name__ == "__main__":
    main()
