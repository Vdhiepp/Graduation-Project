import streamlit as st
import spacy
import pickle
import numpy as np
from PIL import Image
import os

# Load NLP model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

# Load data: encoder, image index, synonyms
@st.cache_resource
def load_data():
    with open("saved_model_GAT/entity_encoder.pkl", "rb") as f:
        entity_encoder = pickle.load(f)
    with open("saved_model_GAT/entity_idx_to_images.pkl", "rb") as f:
        entity_idx_to_images = pickle.load(f)
    with open("saved_model_GAT/synonym_map.pkl", "rb") as f:
        synonym_map = pickle.load(f)
    return entity_encoder, entity_idx_to_images, synonym_map

# Hàm trích xuất triplet đầy đủ
def extract_multiple_triplets_from_caption(caption):
    doc = nlp(caption)
    triplets = []

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text.lower()
                predicate = token.head.text.lower()
                obj = None
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr") and child.pos_ in ("NOUN", "PRON"):
                        obj = child.text.lower()
                        triplets.append((subject, predicate, obj))
                    elif child.dep_ == "prep":
                        pobj = next((t for t in child.children if t.dep_ == "pobj"), None)
                        if pobj:
                            if obj:
                                triplets.append((obj, child.text.lower(), pobj.text.lower()))
                                for conj in pobj.children:
                                    if conj.dep_ == "conj":
                                        triplets.append((obj, child.text.lower(), conj.text.lower()))
                            else:
                                triplets.append((subject, predicate, pobj.text.lower()))

        for token in sent:
            if token.dep_ == "prep" and token.head.pos_ == "NOUN":
                pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
                if pobj:
                    triplets.append((token.head.text.lower(), token.text.lower(), pobj.text.lower()))
                    for conj in pobj.children:
                        if conj.dep_ == "conj":
                            triplets.append((token.head.text.lower(), token.text.lower(), conj.text.lower()))

        for token in sent:
            if token.dep_ == "conj" and token.head.pos_ == "NOUN":
                for parent in token.head.children:
                    if parent.dep_ in ("det", "amod"):
                        triplets.append((token.head.text.lower(), "and", token.text.lower()))

        for token in sent:
            if token.dep_ == "amod" and token.head.pos_ == "NOUN" and token.pos_ == "VERB":
                modifier = next((child for child in token.children if child.dep_ == "npadvmod"), None)
                if modifier:
                    triplets.append((token.head.text.lower(), token.text.lower(), modifier.text.lower()))

        for token in sent:
            if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                triplets.append((token.head.text.lower(), "amod", token.text.lower()))

        for token in sent:
            if token.dep_ == "advcl" and token.pos_ == "VERB":
                verb = token.text.lower()
                subject = None
                obj = None
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child.text.lower()
                    elif child.dep_ in ("dobj", "attr", "pobj") and child.pos_ == "NOUN":
                        obj = child.text.lower()
                if subject:
                    triplets.append((subject, verb, obj))

        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "acl":
                subj = token.head.text.lower()
                verb1 = token.text.lower()
                triplets.append((subj, verb1, None))

                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "to":
                        for grandchild in child.children:
                            if grandchild.dep_ in ("amod", "xcomp") and grandchild.head.pos_ == "NOUN":
                                triplets.append((subj, grandchild.text.lower(), grandchild.head.text.lower()))

        # 8. noun + V-ing + object (e.g., a boy riding a horse)
        for token in sent:
            if token.dep_ == "acl" and token.pos_ == "VERB":
                subject = token.head.text.lower()
                predicate = token.text.lower()
                obj = None
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj") and child.pos_ in ("NOUN", "PROPN", "PRON"):
                        obj = child.text.lower()
                        triplets.append((subject, predicate, obj))

    return triplets

# App interface
st.set_page_config(page_title="Search images from caption", layout="wide")
st.title("Search images from caption")

nlp = load_nlp()
entity_encoder, entity_idx_to_images, synonym_map = load_data()

caption = st.text_input("Enter a caption to search:")

if st.button("Search"):
    if not caption.strip():
        st.warning("Please enter a valid caption.")
    else:
        with st.spinner("Extracting triplets and searching images..."):
            triplets = extract_multiple_triplets_from_caption(caption)

            # Lọc bỏ triplet có None
            triplets = [t for t in triplets if all(t)]

            if not triplets:
                st.info("No valid triplet extracted from the caption.")
            else:
                for t in triplets:
                    st.markdown(f"- ({t[0]}, {t[1]}, {t[2]})")

                def get_ids(word):
                    ids = set()
                    if not word:
                        return ids
                    word_lower = word.lower()
                    syn = synonym_map.get(word_lower)
                    for w in [word_lower, syn]:
                        if w:
                            try:
                                idx = entity_encoder.transform([w])[0]
                                ids.add(idx)
                            except:
                                continue
                    return ids

                def get_images(ids):
                    images = set()
                    for i in ids:
                        images.update(entity_idx_to_images.get(i, []))
                    return images

                for s, p, o in triplets:
                    subj_ids = get_ids(s)
                    pred_ids = get_ids(p)
                    obj_ids = get_ids(o)

                    subj_imgs = get_images(subj_ids)
                    pred_imgs = get_images(pred_ids)
                    obj_imgs = get_images(obj_ids)

                    # Giao cả 3
                    inter_all = subj_imgs & pred_imgs & obj_imgs
                    cols = st.columns(5)
                    for idx, img_id in enumerate(sorted(inter_all)):
                        with cols[idx % 5]:
                            st.image(f"E:/Download/val2017/{int(img_id):012}.jpg", use_container_width=True)

                    # Giao từng cặp
                    inter_sp = subj_imgs & pred_imgs
                    inter_so = subj_imgs & obj_imgs
                    inter_po = pred_imgs & obj_imgs
                    for inter_set in [inter_sp, inter_so, inter_po]:
                        cols = st.columns(5)
                        for idx, img_id in enumerate(sorted(inter_set)):
                            with cols[idx % 5]:
                                st.image(f"E:/Download/val2017/{int(img_id):012}.jpg", use_container_width=True)

                    # Xen kẽ subject – object
                    subj_list = sorted(subj_imgs)
                    obj_list = sorted(obj_imgs)
                    i = j = 0
                    max_total = min(20, len(subj_list) + len(obj_list))
                    cols = st.columns(5)
                    for idx in range(max_total):
                        if i < len(subj_list):
                            with cols[idx % 5]:
                                st.image(f"E:/Download/val2017/{int(subj_list[i]):012}.jpg", use_container_width=True)
                                i += 1
                        if j < len(obj_list) and (i + j) < max_total:
                            with cols[idx % 5]:
                                st.image(f"E:/Download/val2017/{int(obj_list[j]):012}.jpg", use_container_width=True)
                                j += 1
