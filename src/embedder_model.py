from transformers import AutoModel, AutoTokenizer


class Embedder:
    def __init__(self, model_path, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
