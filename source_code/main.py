#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from collections import defaultdict
import random

# === 1. ĐỌC & TIỀN XỬ LÝ DỮ LIỆU ===
df = pd.read_csv("f_coco_triplets.csv")

# Gán chỉ số cho entity
all_entities = pd.concat([df['subject'], df['object']]).unique()
entity_encoder = LabelEncoder()
entity_encoder.fit(all_entities)

df['subject_id'] = entity_encoder.transform(df['subject'])
df['object_id'] = entity_encoder.transform(df['object'])

# Gán chỉ số cho quan hệ
relation_encoder = LabelEncoder()
df['predicate_id'] = relation_encoder.fit_transform(df['predicate'])

# Node features: one-hot cho mỗi entity
num_nodes = len(entity_encoder.classes_)
x = torch.eye(num_nodes)

# Edge index
edge_index = torch.tensor([df['subject_id'].values, df['object_id'].values], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)


# In[19]:


# === 2. MÔ HÌNH GAT ===
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4)
        self.gat2 = GATConv(hidden_channels * 4, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return x


# In[20]:


# === 3. CONTRASTIVE LOSS ===
def contrastive_loss(x_i, x_j, label, margin=1.0):
    dist = F.pairwise_distance(x_i, x_j)
    return (label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)).mean()


# In[30]:


# === 4. HUẤN LUYỆN MÔ HÌNH ===
def train_model(data, model, epochs=100, lr=0.005, samples_per_epoch=2000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_index = data.edge_index.t().tolist()
    all_nodes = list(range(data.num_nodes))
    existing_edges = set(tuple(e) for e in edge_index)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data)

        # Positive samples: từ các cạnh trong đồ thị
        pos_samples = random.sample(edge_index, min(samples_per_epoch, len(edge_index)))

        # Negative samples: node pair không kết nối
        neg_samples = []
        while len(neg_samples) < len(pos_samples):
            i, j = random.sample(all_nodes, 2)
            if (i, j) not in existing_edges and (j, i) not in existing_edges:
                neg_samples.append((i, j))

        # Lấy embedding
        pos_i = torch.stack([embeddings[i] for i, j in pos_samples])
        pos_j = torch.stack([embeddings[j] for i, j in pos_samples])
        neg_i = torch.stack([embeddings[i] for i, j in neg_samples])
        neg_j = torch.stack([embeddings[j] for i, j in neg_samples])

        # Tính loss
        loss_pos = contrastive_loss(pos_i, pos_j, torch.ones(len(pos_i)))
        loss_neg = contrastive_loss(neg_i, neg_j, torch.zeros(len(neg_i)))
        loss = loss_pos + loss_neg

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model


# In[31]:


# === 5. CHẠY HUẤN LUYỆN ===
model = GATModel(in_channels=x.shape[1], hidden_channels=32, out_channels=64)
trained_model = train_model(data, model)


# In[32]:


# === 6. LẤY EMBEDDING CHO CÁC NODE ===
node_embeddings = trained_model(data).detach().numpy()


# In[33]:


# === 7. TẠO BẢN ĐỒ ENTITY → IMAGE ===
entity_idx_to_images = defaultdict(set)
for i, row in df.iterrows():
    entity_idx_to_images[row['subject_id']].add(row['image_id'])
    entity_idx_to_images[row['object_id']].add(row['image_id'])


# In[34]:


# === 8. HÀM TÌM ẢNH LIÊN QUAN ===
def find_related_images(query_image_id, top_k=5):
    related_entities = set(df[df['image_id'] == query_image_id]['subject_id']) | \
                       set(df[df['image_id'] == query_image_id]['object_id'])

    if not related_entities:
        return []

    query_vec = sum(torch.tensor(node_embeddings[i]) for i in related_entities) / len(related_entities)
    scores = cosine_similarity(query_vec.reshape(1, -1), node_embeddings)[0]

    top_entity_indices = scores.argsort()[-top_k:][::-1]

    related_images = set()
    for idx in top_entity_indices:
        related_images.update(entity_idx_to_images[idx])

    return list(related_images - {query_image_id})[:top_k]


# In[35]:


# === 9. HÀM ĐÁNH GIÁ PRECISION / RECALL / F1 ===
def evaluate_retrieval(df, node_embeddings, entity_idx_to_images, top_k=5, sample_size=100):
    image_ids = list(df['image_id'].unique())
    sampled_queries = random.sample(image_ids, min(sample_size, len(image_ids)))

    precision_list, recall_list, f1_list = [], [], []

    for query_id in sampled_queries:
        related_entities = set(df[df['image_id'] == query_id]['subject_id']) | \
                           set(df[df['image_id'] == query_id]['object_id'])

        ground_truth = set()
        for e in related_entities:
            ground_truth.update(entity_idx_to_images[e])
        ground_truth.discard(query_id)

        predicted = set(find_related_images(query_id, top_k=top_k))

        tp = len(predicted & ground_truth)
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "Precision@{}".format(top_k): round(sum(precision_list) / len(precision_list), 4),
        "Recall@{}".format(top_k): round(sum(recall_list) / len(recall_list), 4),
        "F1-score@{}".format(top_k): round(sum(f1_list) / len(f1_list), 4)
    }


# In[36]:


# === 10. CHẠY ĐÁNH GIÁ ===
results = evaluate_retrieval(df, node_embeddings, entity_idx_to_images, top_k=5, sample_size=100)
print("Kết quả đánh giá:")
print(results)


# In[ ]:




