import os
import re
import json
import collections
import logging
import io
from typing import List
from symspellpy import SymSpell, Verbosity
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from charset_normalizer import detect

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("file_processing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DICT_PATH = "vietnamese_words.txt"
ABBR_PATH = "abbreviations.json"

# Hàm phát hiện encoding của file
def detect_file_encoding(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            result = detect(f.read(10000))  # Đọc 10KB đầu để đoán encoding
            encoding = result.get("encoding", "utf-8")
            logger.info(f"Detected encoding for {file_path}: {encoding}")
            return encoding if encoding else "utf-8"
    except Exception as e:
        logger.warning(f"Could not detect encoding for {file_path}: {str(e)}. Falling back to utf-8.")
        return "utf-8"

# ===========================
# Hàm làm sạch cơ bản
# ===========================
def clean_text(text: str):
    text = text.lower()
    greetings = [
        r"\bxin chào\b", r"\bchào\b", r"\bhello+\b", r"\bhi+\b",
        r"\balo+\b", r"\bhey+\b", r"\bchao\b"
    ]
    for g in greetings:
        text = re.sub(g, " ", text, flags=re.IGNORECASE)
        
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
                  r"ỳýỷỹỵđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===========================
# Tokenizer đơn giản cho tiếng Việt
# ===========================
def tokenize_vietnamese(text: str):
    text = clean_text(text)
    return text.split()

# ===========================
# Hàm load dictionary & abbreviations
# ===========================
def load_resources_if_changed():
    global sym_spell, abbreviations, last_dict_mtime, last_abbr_mtime

    # Kiểm tra dictionary
    if os.path.exists(DICT_PATH):
        dict_mtime = os.path.getmtime(DICT_PATH)
        if last_dict_mtime is None or dict_mtime != last_dict_mtime:
            logger.info("Reloading dictionary...")
            sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            encoding = detect_file_encoding(DICT_PATH)
            with open(DICT_PATH, "r", encoding=encoding, errors="ignore") as f:
                dictionary_content = f.read()
            dictionary_stream = io.StringIO(dictionary_content)
            sym.load_dictionary(dictionary_stream, term_index=0, count_index=1, separator=" ")
            sym_spell = sym
            last_dict_mtime = dict_mtime
            logger.info(f"✅ Dictionary loaded with {sym_spell.words_count} words")

    # Kiểm tra abbreviations
    if os.path.exists(ABBR_PATH):
        abbr_mtime = os.path.getmtime(ABBR_PATH)
        if last_abbr_mtime is None or abbr_mtime != last_abbr_mtime:
            logger.info("Reloading abbreviations...")
            encoding = detect_file_encoding(ABBR_PATH)
            with open(ABBR_PATH, "r", encoding=encoding, errors="ignore") as f:
                abbreviations = json.load(f)
            last_abbr_mtime = abbr_mtime
            logger.info(f"✅ Abbreviations loaded with {len(abbreviations)} entries")


# ===========================
# Cập nhật từ điển từ file
# ===========================
def update_dictionary_from_files(file_paths: List[str]):
    counter = collections.Counter()
    supported_extensions = {".txt", ".csv", ".pdf", ".docx"}

    # Load từ điển cũ nếu có
    if os.path.exists(DICT_PATH):
        try:
            encoding = detect_file_encoding(DICT_PATH)
            with open(DICT_PATH, "r", encoding=encoding, errors="ignore") as f:
                for line in f:
                    try:
                        word, freq = line.strip().split()
                        counter[word] = int(freq)
                    except ValueError:
                        logger.warning(f"Invalid line format in dictionary file: {line.strip()}")
                        continue
            logger.info(f"Loaded existing dictionary from {DICT_PATH} with {len(counter)} words")
        except Exception as e:
            logger.error(f"Failed to load dictionary from {DICT_PATH}: {str(e)}")

    # Đọc file và đếm từ
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        
        # Kiểm tra định dạng file trước
        if ext not in supported_extensions:
            logger.warning(f"Unsupported file extension for {path}. Skipping.")
            continue
        
        # Kiểm tra sự tồn tại của file
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            continue

        try:
            if ext == ".txt" or ext == ".csv":
                encoding = detect_file_encoding(path)
                with open(path, "r", encoding=encoding, errors="ignore") as f:
                    # Đọc file theo luồng
                    text = ""
                    for line in f:
                        text += line if ext == ".txt" else " ".join(line.strip().split())
                logger.info(f"Successfully loaded {ext[1:].upper()} file: {path} with encoding {encoding}")
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
                text = " ".join([d.page_content for d in loader.load()])
                logger.info(f"Successfully loaded PDF file: {path}")
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(path)
                text = " ".join([d.page_content for d in loader.load()])
                logger.info(f"Successfully loaded DOCX file: {path}")

            tokens = tokenize_vietnamese(text)
            counter.update(tokens)
            logger.info(f"Processed {len(tokens)} tokens from {path}")

        except Exception as e:
            logger.error(f"Failed to process file {path}: {str(e)}")
            continue

    # Lưu lại từ điển
    try:
        with open(DICT_PATH, "w", encoding="utf-8") as f:
            for word, freq in counter.most_common():
                f.write(f"{word} {freq}\n")
        logger.info(f"Successfully saved dictionary to {DICT_PATH} with {len(counter)} words")
    except Exception as e:
        logger.error(f"Failed to save dictionary to {DICT_PATH}: {str(e)}")

    return {"total_words": len(counter)}

# ===========================
# Load SymSpell
# ===========================
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
if os.path.exists(DICT_PATH):
    try:
        encoding = detect_file_encoding(DICT_PATH)
        with open(DICT_PATH, "r", encoding=encoding, errors="ignore") as f:
            dictionary_content = f.read()
        # Sử dụng io.StringIO để truyền nội dung từ điển dưới dạng stream
        dictionary_stream = io.StringIO(dictionary_content)
        sym_spell.load_dictionary(dictionary_stream, term_index=0, count_index=1, separator=" ")
        logger.info(f"Successfully loaded SymSpell dictionary from {DICT_PATH} with encoding {encoding}")
    except Exception as e:
        logger.error(f"Failed to load SymSpell dictionary from {DICT_PATH}: {str(e)}")

# Load viết tắt
abbreviations = {}
if os.path.exists(ABBR_PATH):
    try:
        with open(ABBR_PATH, "r", encoding=detect_file_encoding(ABBR_PATH), errors="ignore") as f:
            abbreviations = json.load(f)
        logger.info(f"Successfully loaded abbreviations from {ABBR_PATH}")
    except Exception as e:
        logger.error(f"Failed to load abbreviations from {ABBR_PATH}: {str(e)}")

# ===========================
# Hàm chuẩn hóa query
# ===========================
# def normalize_query(text: str):
#     load_resources_if_changed()
#     tokens = tokenize_vietnamese(text)
#     corrected = []

#     for token in tokens:
#         # Nếu token nằm trong từ viết tắt → thay nghĩa và bỏ qua sửa chính tả
#         if token in abbreviations:
#             corrected.append(abbreviations[token])
#         else:
#             # Chỉ sửa chính tả khi không phải viết tắt
#             suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
#             if suggestions:
#                 corrected.append(suggestions[0].term)
#             else:
#                 corrected.append(token)

#     return " ".join(corrected)


# ===========================
# Hàm chuẩn hóa query (chỉ thay viết tắt, không sửa chính tả)
# ===========================
def normalize_query(text: str):
    load_resources_if_changed()  # Kiểm tra & load lại từ điển viết tắt nếu có thay đổi
    tokens = tokenize_vietnamese(text)
    
    # Chỉ thay từ viết tắt
    replaced_tokens = [abbreviations.get(token, token) for token in tokens]

    return " ".join(replaced_tokens)
