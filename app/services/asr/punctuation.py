"""Restore only a missing final mark using FunASR's standard EOF policy."""

from functools import lru_cache

from funasr import AutoModel

from app.core.config import settings
from app.infrastructure import resolve_model_path

MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
_TERMINALS = ".?!\u3002\uff1f\uff01\u2026"
_CLOSING_QUOTES = "\"'\u2019\u201d\u300d\u300f\u3011)]}"
_CONTINUATIONS = ",;:\uff0c\uff1b\uff1a\u3001"
_ENGLISH_MARKS = str.maketrans("\u3002\uff1f\uff01", ".?!")


@lru_cache(maxsize=1)
def get_punctuation_model() -> AutoModel:
    return AutoModel(
        model=resolve_model_path(MODEL_ID),
        device="cpu",
        **settings.FUNASR_AUTOMODEL_KWARGS,
    )


def restore_sentence_ending(text: str) -> str:
    ending = text.rstrip().rstrip(_CLOSING_QUOTES).rstrip()
    if not ending or ending[-1] in _TERMINALS:
        return text
    results = get_punctuation_model().generate(input=text)
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
        or not isinstance(results[0].get("text"), str)
        or not results[0]["text"].strip()
    ):
        raise RuntimeError("Punctuation model returned no valid text")
    restored = results[0]["text"].rstrip().rstrip(_CLOSING_QUOTES).rstrip()
    if not restored or restored[-1] not in _TERMINALS:
        return text
    # FunASR includes EOF period restoration; never copy its interior rewrites.
    mark = restored[-1]
    if ending[-1].isascii() and ending[-1].isalnum():
        mark = mark.translate(_ENGLISH_MARKS)
    suffix = text[len(ending) :]
    if ending[-1] in _CONTINUATIONS:
        ending = ending[:-1]
    return ending + mark + suffix
