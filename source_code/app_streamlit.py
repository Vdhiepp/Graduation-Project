import streamlit as st
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import Counter
import spacy

st.set_page_config(
    page_title="Tìm ảnh từ Caption",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load spaCy và dữ liệu
nlp = spacy.load("en_core_web_sm")

with open("saved_model_R_GCN/entity_encoder.pkl", "rb") as f:
    entity_encoder = pickle.load(f)
with open("saved_model_R_GCN/entity_idx_to_images.pkl", "rb") as f:
    entity_idx_to_images = pickle.load(f)
with open("saved_model_R_GCN/synonym_map.pkl", "rb") as f:
    synonym_map = pickle.load(f)

# Hàm trích triplet
def extract_multiple_triplets_from_caption(caption):
    doc = nlp(caption)
    subjects, predicate, objects = [], None, []
    noun_candidates = [t for t in doc if t.pos_ == "NOUN"]
    if noun_candidates:
        first_noun = noun_candidates[0]
        subjects.append(first_noun.text.lower())
        for token in noun_candidates[1:]:
            if token.dep_ == "conj" or token.head == first_noun:
                subjects.append(token.text.lower())
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            predicate = token.text.lower()
            break
    if predicate is None:
        for token in doc:
            if token.dep_ == "acl" and token.pos_ == "VERB":
                predicate = token.text.lower()
                break
    for token in doc:
        if token.dep_ in ("dobj", "pobj", "attr") and token.pos_ == "NOUN":
            objects.append(token.text.lower())
        elif token.dep_ == "conj" and token.head.dep_ in ("dobj", "pobj", "attr"):
            objects.append(token.text.lower())
    if predicate is None and not objects:
        for token in doc:
            if token.dep_ == "prep":
                pobj = next((t for t in token.children if t.dep_ == "pobj" and t.pos_ == "NOUN"), None)
                if pobj:
                    objects.append(pobj.text.lower())
    triplets = []
    if subjects and objects:
        for subj in subjects:
            for obj in objects:
                triplets.append((subj, predicate, obj))
    elif subjects:
        triplets = [(subj, predicate, None) for subj in subjects]
    return triplets

# Hàm tìm kiếm ảnh
def find_images_by_entities_prioritize_intersection(caption, entity_encoder, entity_idx_to_images, synonym_map):
    triplets = extract_multiple_triplets_from_caption(caption)

    def normalize(word):
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

    def get_id(word, label, raw_word):
        if word:
            try:
                idx = entity_encoder.transform([word])[0]
                return idx
            except:
                pass
        return None

    image_counter = Counter()
    if not triplets:
        doc = nlp(caption)
        verb, noun = None, None
        for token in doc:
            if token.pos_ == "VERB" and not verb:
                verb = normalize(token.text)
            elif token.pos_ == "NOUN" and not noun:
                noun = normalize(token.text)
        pred_id = get_id(verb, "predicate (fallback)", verb)
        obj_id = get_id(noun, "object (fallback)", noun)
        imgs_riding = set(entity_idx_to_images.get(pred_id, [])) if pred_id is not None else set()
        imgs_object = set(entity_idx_to_images.get(obj_id, [])) if obj_id is not None else set()
        inter_imgs = imgs_riding & imgs_object
        imgs = inter_imgs if inter_imgs else imgs_riding | imgs_object
        if not imgs:
            return []
        return [f"{int(img_id):012}.jpg" for img_id in imgs]

    for subj_raw, pred_raw, obj_raw in triplets:
        subj = normalize(subj_raw)
        pred = normalize(pred_raw)
        obj = normalize(obj_raw)
        subj_id = get_id(subj, "subject", subj_raw)
        pred_id = get_id(pred, "predicate", pred_raw)
        obj_id = get_id(obj, "object", obj_raw)
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
            subj_imgs = set(entity_idx_to_images.get(subj_id, []))
            pred_imgs = set(entity_idx_to_images.get(pred_id, []))
            imgs = subj_imgs & pred_imgs or subj_imgs | pred_imgs
        elif obj_id is not None and pred_id is not None:
            obj_imgs = set(entity_idx_to_images.get(obj_id, []))
            pred_imgs = set(entity_idx_to_images.get(pred_id, []))
            imgs = obj_imgs & pred_imgs or obj_imgs | pred_imgs
        elif subj_id is not None:
            imgs = set(entity_idx_to_images.get(subj_id, []))
        elif obj_id is not None:
            imgs = set(entity_idx_to_images.get(obj_id, []))
        image_counter.update(imgs)

    if not image_counter:
        return []
    sorted_image_ids = [img_id for img_id, _ in image_counter.most_common()]
    return [f"{int(img_id):012}.jpg" for img_id in sorted_image_ids]

# Giao diện Streamlit
st.title("Search images from caption")

caption = st.text_input("Enter image description (Example: 'a man riding a horse')")

if caption:
    st.subheader("Triplets trích được từ caption")
    triplets = extract_multiple_triplets_from_caption(caption)
    if triplets:
        for s, p, o in triplets:
            st.write(f"- ({s}, {p}, {o})")
    else:
        st.warning("Could not extract any triplets.")

if st.button("Search"):
    results = find_images_by_entities_prioritize_intersection(
        caption=caption,
        entity_encoder=entity_encoder,
        entity_idx_to_images=entity_idx_to_images,
        synonym_map=synonym_map
    )

    if results:
        st.success(f"Found {len(results)} related images.")
        for row_start in range(0, len(results), 3):
            cols = st.columns(3)
            for i in range(3):
                if row_start + i < len(results):
                    img_name = results[row_start + i]
                    img_path = os.path.join("E:/Download/val2017", img_name)
                    if os.path.exists(img_path):
                        with cols[i]:
                            st.image(Image.open(img_path), caption=img_name, use_container_width=True)
                    else:
                        with cols[i]:
                            st.error(f"Image file not found {img_name}")
    else:
        st.warning("No matching image found.")
