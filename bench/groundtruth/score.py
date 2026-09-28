import re

import jiwer

NUM_WORDS = [
    "ноль",
    "нуля",
    "нулю",
    "нулем",
    "один",
    "одного",
    "одному",
    "одним",
    "одном",
    "одна",
    "одной",
    "одну",
    "одно",
    "два",
    "две",
    "двух",
    "двум",
    "двумя",
    "три",
    "трех",
    "трем",
    "тремя",
    "четыре",
    "четырех",
    "четырем",
    "четырьмя",
    "пять",
    "пяти",
    "пятью",
    "шесть",
    "шести",
    "шестью",
    "семь",
    "семи",
    "семью",
    "восемь",
    "восьми",
    "восемью",
    "девять",
    "девяти",
    "девятью",
    "десять",
    "десяти",
    "десятью",
    "сорок",
    "сорока",
    "пятьдесят",
    "пятидесяти",
    "шестьдесят",
    "шестидесяти",
    "семьдесят",
    "семидесяти",
    "восемьдесят",
    "восьмидесяти",
    "девяносто",
    "девяноста",
    "сто",
    "ста",
    "сотни",
    "сотен",
    "двести",
    "двухсот",
    "триста",
    "трехсот",
    "четыреста",
    "четырехсот",
    "пятьсот",
    "пятисот",
    "шестьсот",
    "шестисот",
    "семьсот",
    "семисот",
    "восемьсот",
    "восьмисот",
    "девятьсот",
    "девятисот",
    "полтора",
    "полторы",
]
NUM_RE = re.compile(
    r"^(\d+|(одиннадцат|двенадцат|тринадцат|четырнадцат|пятнадцат|шестнадцат|семнадцат|восемнадцат|девятнадцат|двадцат|тридцат)(ь|и|ью)|тысяч\w*|миллион\w*|миллиард\w*)$"
)
NUMS = set(NUM_WORDS)


def normalize(t: str) -> str:
    t = (t or "").lower().replace("ё", "е")
    t = re.sub(r"(\d)[.,](\d)", r"\1\2", t)  # 1,5 / 3.14 -> one digit token
    t = re.sub(r"(\d+)[-–]?[а-я]{1,3}\b", r"\1", t)  # 22-го, 90-х -> 22, 90
    t = re.sub(r"[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_num(t: str) -> str:
    """Number-insensitive: every maximal run of digit tokens / cardinal numeral words becomes one '<n>' token."""
    out = []
    for w in normalize(t).split():
        isnum = w in NUMS or bool(NUM_RE.match(w))
        if isnum:
            if not out or out[-1] != "<n>":
                out.append("<n>")
        else:
            out.append(w)
    return " ".join(out)


def wer_cer(refs, hyps, fn=normalize):
    r = [fn(x) for x in refs]
    h = [fn(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    r = [r[i] for i in keep]
    h = [h[i] for i in keep]
    return jiwer.wer(r, h), jiwer.cer(r, h), sum(len(x.split()) for x in r)
