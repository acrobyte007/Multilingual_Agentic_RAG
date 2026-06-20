import os
import gc
import torch
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from logger.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAVE_DIR = "./miniLM_model"
ONNX_DIR = "./onnx_model"
QUANT_MODEL_PATH = "./onnx_model/model_quantized.onnx"
device = "cpu"
logger.info("Loading SentenceTransformer model...")
st_model = SentenceTransformer(MODEL_NAME)
st_model.save(SAVE_DIR)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
logger.info("Original model saved.")
logger.info("Exporting to ONNX...")
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    SAVE_DIR,
    export=True
)
ort_model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)
logger.info("ONNX model saved.")
logger.info("Quantizing model...")
onnx_fp32 = os.path.join(ONNX_DIR, "model.onnx")
quantize_dynamic(
    model_input=onnx_fp32,
    model_output=QUANT_MODEL_PATH,
    weight_type=QuantType.QInt8
)

logger.info("Quantization complete.")
del st_model
del ort_model
gc.collect()
torch.cuda.empty_cache()
logger.info("Original model removed from RAM.")
quant_model = ORTModelForFeatureExtraction.from_pretrained(
    ONNX_DIR,
    file_name="model_quantized.onnx"
)
tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
logger.info("Quantized model loaded.")

def tokenize_sentences(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
    encoded = tokenizer(sentences, padding=True, truncation=True)
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
    with torch.no_grad():
        outputs = quant_model(**batch)
        attention_mask = batch["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state
        masked = hidden * attention_mask
        embeddings = masked.sum(dim=1) / attention_mask.sum(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().tolist()
    result = {}
    for i in range(len(sentences)):
        result[i] = {
            "tokens": tokens_list[i],
            "embedding": embeddings[i]
        }
    return result

