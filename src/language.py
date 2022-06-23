"""
Module for language detection
"""
import langdetect


class AbstractLanguageDetection:
    def detect(self, text: str):
        raise NotImplementedError("Attempted to invoke abstract language detection")

    def is_greek(self, lang: str):
        raise NotImplementedError("Attempted to invoke abstract language detection")


class LangDetect(AbstractLanguageDetection):
    """
    Detection using langdetect
    https://github.com/Mimino666/langdetect
    """

    def detect(self, text: str):
        return langdetect.detect(text)

    def is_greek(self, language: str = None):
        return language == "el"
