import logging
import time
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
model.to(device)
model.eval()

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def tokenize_sentences(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True
    )
    tokens_list = [
        tokenizer.convert_ids_to_tokens(ids)
        for ids in encoded["input_ids"]
    ]
    return tokens_list


def embedding_process(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
    tokens_list = tokenize_sentences(sentences)
    batch = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().tolist()
    result = {}
    for i in range(len(sentences)):
        result[i] = {
            "tokens": tokens_list[i],
            "embedding": embeddings[i]
        }
    return result