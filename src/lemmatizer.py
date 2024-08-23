import re
from abc import ABC, abstractmethod

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc


class LemmatizerInterface(ABC):
    @abstractmethod
    def lemmatize_text(
        self,
        text: str,
    ) -> str:
        pass

class NatashaBasedLemmatizer(LemmatizerInterface):
    def __init__(
        self,
    ):
        super().__init__()

        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)
        self._syntax_parser = NewsSyntaxParser(self._emb)

    def lemmatize_text(
        self,
        text: str,
    ) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)

        tokens_lemmatized: list[str] = [token.lemma for token in doc.tokens]

        return ' '.join(tokens_lemmatized)
