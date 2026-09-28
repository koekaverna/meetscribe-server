"""Known Whisper hallucinations on silence and noise.

Whisper was trained on subtitled video, so on non-speech it emits subtitle credits and video
sign-offs. With large-v3-turbo these come with no_speech_prob 0.0 and an ordinary avg_logprob
(measured on a real meeting: 48 of 393 chunk segments, median avg_logprob -0.28), so no
confidence threshold separates them from speech. Only the text does.

The list is deliberately narrow: phrases that occur in video subtitles and not in a meeting.
"Спасибо за внимание" and "Добро пожаловать" are hallucinated too, but people do say them.
"""

import re

_PATTERNS = (
    r"(редактор )?субтитр\w*( \w+){0,7}",
    r"корректор( \w+){1,2}",
    r"продолжение следует",
    r"спасибо за просмотр",
    r"подписывайтесь( на( наш| мой)? канал)?",
    r"с вами был игорь негода",
    r"увидимся в следующ(ем|их) видео",
    r"до новых встреч",
    r"фондю любит тебя",
    r"конец (видео|воды|фильма)",
    r"продолжение воды",
    r"(спокойная|динамичная|бодрая|тревожная|веселая|грустная|лирическая|напряженная) музыка",
)
_KNOWN = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))
_NOT_WORD = re.compile(r"[^\w\s]+")


def normalize(text: str) -> str:
    return " ".join(_NOT_WORD.sub(" ", text.lower().replace("ё", "е")).split())


def is_known_hallucination(text: str) -> bool:
    """True when the whole segment is one known hallucinated phrase, optionally repeated."""
    words = normalize(text)
    if not words:
        return False
    rest = words
    while rest:
        m = _KNOWN.match(rest)
        if m is None or m.end() == 0:
            return False
        if m.end() < len(rest) and rest[m.end()] != " ":
            return False
        rest = rest[m.end() :].lstrip()
    return True
