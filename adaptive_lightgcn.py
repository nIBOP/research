import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from torch.nn.utils import clip_grad_norm_

class CustomTrainer(Trainer):
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """Переопределяем метод для вывода логов батчей вместо progress bar"""
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        
        batch_interval = 50  # Интервал печати логов
        total_batches = len(train_data)
        
        for batch_idx, interaction in enumerate(train_data):
            # ВАЖНО: нужно обновлять внутренние структуры RecBole, иначе он будет сэмплировать одно и то же!
            if hasattr(train_data, 'pr_end') and batch_idx == total_batches - 1:
                train_data.pr_end() # Говорим даталоадеру, что эпоха закончилась и нужно сделать shuffle/resample

            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            
            # Обработка кортежа потерь
            if isinstance(losses, tuple):
                loss = losses[0]  # Суммарный лосс для backward
                loss_bpr = losses[1].item()
                loss_sem = losses[2].item()
                
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                loss_bpr = loss.item()
                loss_sem = 0.0
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                
            self._check_nan(loss)
            loss.backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
                
            # --- НАШИ ЛОГИ: выводим без засорения памяти ---
            if batch_idx > 0 and batch_idx % batch_interval == 0:
                print(f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f} (BPR: {loss_bpr:.4f}, Sem: {loss_sem:.4f})")
        
        # Красивый вывод с учетом того, что total_loss может быть кортежем
        if isinstance(total_loss, tuple):
            print(f"🏁 [Epoch {epoch_idx}] Завершена! Суммарный лосс: {total_loss[0]:.4f} (BPR: {total_loss[1]:.4f}, Sem: {total_loss[2]:.4f})")
        else:
            print(f"🏁 [Epoch {epoch_idx}] Завершена! Суммарный лосс: {total_loss:.4f}")
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        """Переопределяем _valid_epoch, чтобы выводить результаты валидации"""
        valid_score, valid_result = super()._valid_epoch(valid_data, show_progress=False)
        print(f"📈 [Validation] Метрики: {valid_result}")
        return valid_score, valid_result

class AdaptiveLightGCN(LightGCN):
    def __init__(self, config, dataset):
        super(AdaptiveLightGCN, self).__init__(config, dataset)
        self.centroids_path = config['centroids_path']
        self.cl_weight = config['proto_reg_weight']
        self.tau = config['temperature']
        self.item_mapping_path = 'clean_movies/clean_movies.item'


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
        # ВАЖНО: Очищаем кэш из LightGCN, иначе метрики будут стоять на месте!
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

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

        current_item_embs = F.normalize(pos_embeddings.detach(), dim=1)
        all_centroids_matrix = F.normalize(all_centroids_matrix, dim=1)

        cos_sim = torch.matmul(current_item_embs, all_centroids_matrix.t())
        logits = cos_sim / self.tau

        proto_loss_vector = F.cross_entropy(logits, cluster_ids, reduction='none')

        batch_alphas = self.item_alpha_weights[pos_item]
        weighted_proto_loss_mean = (batch_alphas * proto_loss_vector).mean()
        
        sem_loss_value = self.cl_weight * weighted_proto_loss_mean

        # Возвращаем tuple, чтобы Trainer мог получить доступ к отдельным компонентам лосса
        return loss_bpr + sem_loss_value, loss_bpr, sem_loss_value