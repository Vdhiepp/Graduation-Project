import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from PIL import Image
import numpy as np
import os
from collections import Counter
import spacy
import pickle

# 1. Load mô hình Detectron2 để detect object
def load_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  
    predictor = DefaultPredictor(cfg)
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return predictor, class_names

# 2. Load mô hình BLIP sinh caption
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# 3. Load dữ liệu R-GCN đã huấn luyện
@st.cache_resource
def load_support_files():
    with open("saved_model_R_GCN/entity_encoder.pkl", "rb") as f:
        entity_encoder = pickle.load(f)
    with open("saved_model_R_GCN/entity_idx_to_images.pkl", "rb") as f:
        entity_idx_to_images = pickle.load(f)
    with open("saved_model_R_GCN/synonym_map.pkl", "rb") as f:
        synonym_map = pickle.load(f)
    return entity_encoder, entity_idx_to_images, synonym_map

# 4. spaCy
nlp = spacy.load("en_core_web_sm")

# 5. Function 
def extract_triplets(caption, labels):
    triplets = []
    subject = "person" if "person" in labels else (labels[0] if labels else None)
    predicate = caption.split()[1] if len(caption.split()) > 1 else "interacts"

    if subject:
        for obj in labels:
            if obj != subject:
                triplets.append((subject, predicate, obj))
    return triplets

# Hàm cải tiến: Trích triplet bằng NLP nếu thiếu object từ detectron2
def extract_multiple_triplets_from_caption(caption):
    doc = nlp(caption)
    subjects, predicate, objects = [], None, []

    # 1. Xác định subject (danh từ đầu tiên hoặc 'man', 'woman'...)
    for token in doc:
        if token.pos_ == "NOUN" and token.dep_ in ("nsubj", "attr", "pobj", "nmod", "ROOT"):
            subjects.append(token.text.lower())
            break

    # fallback subject
    if not subjects:
        for token in doc:
            if token.pos_ == "NOUN":
                subjects.append(token.text.lower())
                break

    # 2. Xác định predicate (động từ chính hoặc giới từ nếu không có động từ)
    for token in doc:
        if token.pos_ == "VERB":
            predicate = token.text.lower()
            break

    # fallback nếu không có verb → dùng giới từ hoặc mặc định 'is'
    if predicate is None:
        for token in doc:
            if token.pos_ == "ADP":  # giới từ như "in", "on", "with"...
                predicate = token.text.lower()
                break
    if predicate is None:
        predicate = "is"

    # 3. Xác định object (tân ngữ hoặc danh từ phía sau giới từ)
    for token in doc:
        if token.dep_ in ("dobj", "pobj", "attr", "conj") and token.pos_ == "NOUN":
            objects.append(token.text.lower())
    if not objects:
        # fallback: tìm bất kỳ noun khác với subject
        for token in doc:
            if token.pos_ == "NOUN" and token.text.lower() not in subjects:
                objects.append(token.text.lower())

    # 4. Sinh triplet
    triplets = []
    if subjects and objects:
        for subj in subjects:
            for obj in objects:
                triplets.append((subj, predicate, obj))
    elif subjects:
        triplets = [(subj, predicate, None) for subj in subjects]
    return triplets

def normalize(word, entity_encoder, synonym_map):
    if not word:
        return None
    word_lower = word.lower()
    word_norm = synonym_map.get(word_lower, word_lower)
    if word_norm in entity_encoder.classes_:
        return word_norm
    if word_norm.endswith("ing"):
        root = word_norm[:-3]
        if root in entity_encoder.classes_:
            return root
    lemma = nlp(word_norm)[0].lemma_
    if lemma in entity_encoder.classes_:
        return lemma
    return word_norm

def get_id(word, entity_encoder, label, raw_word):
    if word:
        try:
            idx = entity_encoder.transform([word])[0]
            return idx
        except:
            pass
    return None

def find_images(triplets, entity_encoder, entity_idx_to_images, synonym_map):
    image_counter = Counter()
    for subj_raw, pred_raw, obj_raw in triplets:
        subj = normalize(subj_raw, entity_encoder, synonym_map)
        pred = normalize(pred_raw, entity_encoder, synonym_map)
        obj = normalize(obj_raw, entity_encoder, synonym_map)

        subj_id = get_id(subj, entity_encoder, "subject", subj_raw)
        pred_id = get_id(pred, entity_encoder, "predicate", pred_raw)
        obj_id = get_id(obj, entity_encoder, "object", obj_raw)

        imgs = set()
        if subj_id is not None and obj_id is not None:
            subj_imgs = set(entity_idx_to_images.get(subj_id, []))
            obj_imgs = set(entity_idx_to_images.get(obj_id, []))
            core_imgs = subj_imgs & obj_imgs
            if pred_id is not None:
                pred_imgs = set(entity_idx_to_images.get(pred_id, []))
                imgs = core_imgs & pred_imgs or core_imgs
            else:
                imgs = core_imgs
        elif subj_id is not None and pred_id is not None:
            imgs = set(entity_idx_to_images.get(subj_id, [])) & set(entity_idx_to_images.get(pred_id, []))
        elif obj_id is not None:
            imgs = set(entity_idx_to_images.get(obj_id, []))
        image_counter.update(imgs)

    sorted_image_ids = [img_id for img_id, _ in image_counter.most_common()]
    return [f"{int(img_id):012}.jpg" for img_id in sorted_image_ids]

# 5. Giao diện Streamlit
st.set_page_config(
    page_title="Tìm ảnh từ Ảnh",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Truy vấn ảnh từ hình ảnh đầu vào")

uploaded_file = st.file_uploader("Chọn một ảnh để truy vấn", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã chọn", use_container_width=True)

    with st.spinner("Đang xử lý..."):
        # Load models
        processor, blip_model = load_blip()
        predictor, class_names = load_detectron2()
        entity_encoder, entity_idx_to_images, synonym_map = load_support_files()

        # BLIP caption
        inputs = processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.subheader("Caption:")
        st.write(caption)

        # Detect objects
        outputs = predictor(np.array(image))
        label_ids = outputs["instances"].pred_classes.tolist()
        labels = list(set([class_names[i] for i in label_ids]))
        st.subheader("Các đối tượng phát hiện:")
        st.write(", ".join(labels))

        # Extract triplets: dùng Detectron2 nếu có labels, nếu không thì dùng NLP
        if labels:
            triplets = extract_triplets(caption, labels)
        else:
            triplets = extract_multiple_triplets_from_caption(caption)

        # Hiển thị triplets
        st.subheader("Triplets trích xuất:")
        if triplets:
            for t in triplets:
                st.write(f"- {t}")
        else:
            st.warning("Không trích xuất được triplet nào từ caption.")

        # Find images
        result_images = find_images(triplets, entity_encoder, entity_idx_to_images, synonym_map)
        if result_images:
            st.success(f"Tìm thấy {len(result_images)} ảnh phù hợp.")
            for i in range(0, len(result_images), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(result_images):
                        img_path = os.path.join("E:/Download/val2017", result_images[i + j])
                        if os.path.exists(img_path):
                            with cols[j]:
                                st.image(Image.open(img_path), caption=result_images[i + j], use_container_width=True)
        else:
            st.warning("Không tìm thấy ảnh nào phù hợp.")
