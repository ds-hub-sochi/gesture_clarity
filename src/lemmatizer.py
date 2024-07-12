import re
from abc import ABC, abstractmethod

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc


class LemmatizerInterface(ABC):
    @abstractmethod
    def lemmatize_text(self, text: str) -> str:
        pass

class NatashaBasedLemmatizer(LemmatizerInterface):
    def __init__(
        self,
    ):
        self._segmenter: natasha.segment.Segmenter = Segmenter()
        self._morph_vocab: natasha.morph.vocab.MorphVocab = MorphVocab()
        self._emb: natasha.emb.NewsEmbedding = NewsEmbedding()
        self._morph_tagger: natasha.morph.tagger.NewsMorphTagger = NewsMorphTagger(self._emb)
        self._syntax_parser: natasha.syntax.NewsSyntaxParser = NewsSyntaxParser(self._emb)

    def lemmatize_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        doc: natasha.doc.Doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)

        tokens_lemmatized: list[str] = [token.lemma for token in doc.tokens]

        return ' '.join(tokens_lemmatized)
