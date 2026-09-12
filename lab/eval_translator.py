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
    val = build_pairs()[:N_VAL]
    en, ru = [p[0] for p in val], [p[1] for p in val]
    for direction, src, ref, st, tt in (("en2ru", en, ru, en_tok, ru_tok), ("ru2en", ru, en, ru_tok, en_tok)):
        model = load(models, direction, st, tt)
        t0 = time.time()
        hyp = [translate(model, s, st, tt) for s in src]
        bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
        chrf = sacrebleu.corpus_chrf(hyp, [ref]).score
        exact = sum(h.strip() == r.strip() for h, r in zip(hyp, ref)) / len(ref)
        print(f"{direction}: BLEU {bleu:.1f} | chrF {chrf:.1f} | дословно совпало {exact:.0%} | {time.time() - t0:.0f} с")
        for s, h, r in list(zip(src, hyp, ref))[:3]:
            print(f"    {s} → {h}   (эталон: {r})")


if __name__ == "__main__":
    main()
