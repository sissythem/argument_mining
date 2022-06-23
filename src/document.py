from src.utils import make_segment_id


class Document:
    @staticmethod
    def make_dummy_document(text):
        return {
            "id": make_segment_id(),
            "content": text,
            "title": "Τίτλος κειμένου"
        }
