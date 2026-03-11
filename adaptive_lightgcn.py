import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from recbole.model.general_recommender.lightgcn import LightGCN

class VerboseDataLoader:
    def __init__(self, dataloader, interval=50):
        self.dataloader = dataloader
        self.interval = interval
        self.total = len(dataloader) if hasattr(dataloader, '__len__') else "?"

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i > 0 and i % self.interval == 0:
                print(f"    ... batch {i}/{self.total}")
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

class AdaptiveLightGCN(LightGCN):
    def __init__(self, config, dataset):
        super(AdaptiveLightGCN, self).__init__(config, dataset)
        self.centroids_path = config.get('centroids_path', 'cluster_centroids.pt')
        self.item_mapping_path = 'clean_movies/clean_movies.item'

        self.cl_weight = config.get('proto_reg_weight', 0.1)
        self.tau = config.get('temperature', 0.5)


        try:
            self.semantic_centroids = torch.load(self.centroids_path).to(self.device)
            self.n_clusters = self.semantic_centroids.shape[0]
            centroid_dim = self.semantic_centroids.shape[1]

            if centroid_dim != self.latent_dim:
                self.centroid_proj = nn.Sequential(
                    nn.Linear(centroid_dim, self.latent_dim * 2),
                    nn.LayerNorm(self.latent_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.latent_dim * 2, self.latent_dim)
                )
            else:
                self.centroid_proj = None
            print(f"[AdaptiveLightGCN] ✅ Loaded {self.n_clusters} semantic centroids.")
        except Exception as e:
            print(f"[AdaptiveLightGCN] ⚠️ Failed to load centroids: {e}")
            self.semantic_centroids = None

        if self.semantic_centroids is not None:
            self.item2cluster = torch.zeros(self.n_items, dtype=torch.long, device=self.device)
            try:
                # 🚨 ИСПРАВЛЕНИЕ ДО НЕГО: Жесткий тип str спасает от бага .0 во float
                df_map = pd.read_csv(self.item_mapping_path, sep='\t', dtype=str)
                token2id = dataset.field2token_id[dataset.iid_field]

                mapped_count = 0
                for _, row in df_map.iterrows():
                    token = row['item_id:token']
                    cluster_idx = int(row['cluster_id:token'])
                    if token in token2id:
                        iid = token2id[token]
                        self.item2cluster[iid] = cluster_idx
                        mapped_count += 1
                print(f"[AdaptiveLightGCN] 🔗 Успешно смапплено {mapped_count} фильмов с кластерами.")
            except Exception as e:
                print(f"[AdaptiveLightGCN] ⚠️ Mapping failed: {e}")

        train_item_ids = dataset.inter_feat[dataset.iid_field]
        item_counts = torch.bincount(train_item_ids, minlength=self.n_items).float().to(self.device)
        self.item_alpha_weights = 1.0 / torch.log(torch.e + item_counts)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Сырые веса (0-й слой)
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=getattr(self, 'require_pow', False)
        )

        loss_bpr = mf_loss + self.reg_weight * reg_loss

        if self.semantic_centroids is None:
            return loss_bpr

        # --- Adaptive Contrastive Loss ---
        cluster_ids = self.item2cluster[pos_item]

        all_centroids_matrix = self.semantic_centroids
        if self.centroid_proj:
            all_centroids_matrix = self.centroid_proj(all_centroids_matrix)

        current_item_embs = F.normalize(pos_embeddings, dim=1)
        all_centroids_matrix = F.normalize(all_centroids_matrix, dim=1)

        cos_sim = torch.matmul(current_item_embs, all_centroids_matrix.t())
        logits = cos_sim / self.tau

        proto_loss_vector = F.cross_entropy(logits, cluster_ids, reduction='none')

        batch_alphas = self.item_alpha_weights[pos_item]
        weighted_proto_loss_mean = (batch_alphas * proto_loss_vector).mean()
        
        sem_loss_value = self.cl_weight * weighted_proto_loss_mean
        
        # 2. РАДАР ГРАДИЕНТОВ: Печатаем лоссы случайным образом (~1% батчей)
        if torch.rand(1).item() < 0.01:
            print(f"⚖️ Баланс: BPR={loss_bpr.item():.4f} | SemLoss={sem_loss_value.item():.4f}")

        return loss_bpr + sem_loss_value