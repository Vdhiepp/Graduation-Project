import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from collections import Counter
import spacy

st.set_page_config(page_title="Image Search from Caption", layout="wide")
nlp = spacy.load("en_core_web_sm")

# Load models
MODELS = {
    "R-GCN": "saved_model_TransE",
    "GAT": "saved_model_GAT",
    "GCN": "saved_model_GCN",
    "TransE": "saved_model_R_GCN"
}

model_data = {}
for name, path in MODELS.items():
    with open(f"{path}/entity_encoder.pkl", "rb") as f:
        entity_encoder = pickle.load(f)
    with open(f"{path}/entity_idx_to_images.pkl", "rb") as f:
        entity_idx_to_images = pickle.load(f)
    with open(f"{path}/synonym_map.pkl", "rb") as f:
        synonym_map = pickle.load(f)
    model_data[name] = {
        "encoder": entity_encoder,
        "idx_to_images": entity_idx_to_images,
        "syn_map": synonym_map
    }

# User-defined functions
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

    def get_id(word):
        if word:
            try:
                return entity_encoder.transform([word])[0]
            except:
                return None
        return None

    image_counter = Counter()
    fallback_counter = Counter()

    if not triplets:
        return []

    for subj_raw, pred_raw, obj_raw in triplets:
        subj = normalize(subj_raw)
        pred = normalize(pred_raw)
        obj = normalize(obj_raw)
        subj_id = get_id(subj)
        pred_id = get_id(pred)
        obj_id = get_id(obj)

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

        if not imgs:
            if subj_id is not None:
                fallback_counter.update(entity_idx_to_images.get(subj_id, []))
            if pred_id is not None:
                fallback_counter.update(entity_idx_to_images.get(pred_id, []))
            if obj_id is not None:
                fallback_counter.update(entity_idx_to_images.get(obj_id, []))

        image_counter.update(imgs)

    if not image_counter:
        image_counter.update(fallback_counter)

    if not image_counter:
        return []

    sorted_image_ids = [img_id for img_id, _ in image_counter.most_common()]
    return [f"{int(img_id):012}.jpg" for img_id in sorted_image_ids]

# UI
st.title("Search images from caption using 4 models")

caption = st.text_input("Enter an image description (e.g., 'a man riding a horse')")

if caption:
    st.subheader("Extracted triplets")
    triplets = extract_multiple_triplets_from_caption(caption)
    if triplets:
        for s, p, o in triplets:
            st.write(f"- ({s}, {p}, {o})")
    else:
        st.warning("No triplets extracted from the caption.")

if st.button("Search"):
    for model_name in MODELS.keys():
        st.subheader(f"Results from {model_name} model")
        model = model_data[model_name]

        results = find_images_by_entities_prioritize_intersection(
            caption,
            entity_encoder=model["encoder"],
            entity_idx_to_images=model["idx_to_images"],
            synonym_map=model["syn_map"]
        )

        if results:
            st.success(f"Found {len(results)} related images.")
            for i in range(0, len(results), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(results):
                        img_name = results[i + j]
                        img_path = os.path.join("E:/Download/val2017", img_name)
                        if os.path.exists(img_path):
                            cols[j].image(Image.open(img_path), caption=img_name, use_container_width=True)
                        else:
                            cols[j].error(f"Image not found: {img_name}")
        else:
            st.warning("No matching images found.")