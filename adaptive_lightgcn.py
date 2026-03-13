import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from torch.nn.utils import clip_grad_norm_


def _cfg(config, key, default):
    """Safely fetch optional config value with a default fallback."""
    try:
        return config[key]
    except KeyError:
        return default

class CustomTrainer(Trainer):
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """Переопределяем метод для вывода логов батчей вместо progress bar"""
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        
        # Накапливаем лоссы как тензоры на GPU!
        total_loss_tensor = torch.tensor(0.0, device=self.model.device)
        total_bpr_tensor = torch.tensor(0.0, device=self.model.device)
        total_sem_tensor = torch.tensor(0.0, device=self.model.device)
        
        batch_interval = 50 
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
                # Плюсуем тензоры напрямую на видеокарте (избегая In-place ошибок broadcast)
                total_loss_tensor = total_loss_tensor + losses[0].detach().squeeze()
                total_bpr_tensor = total_bpr_tensor + losses[1].detach().squeeze()
                total_sem_tensor = total_sem_tensor + losses[2].detach().squeeze()
            else:
                loss = losses
                total_loss_tensor = total_loss_tensor + losses.detach().squeeze()
                total_bpr_tensor = total_bpr_tensor + losses.detach().squeeze()
                
            self._check_nan(loss)
            loss.backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
                
            # Синхронизация CPU-GPU происходит ТОЛЬКО раз в 50 батчей
            if batch_idx > 0 and batch_idx % batch_interval == 0:
                if isinstance(losses, tuple):
                    print(f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | Loss: {losses[0].item():.4f} (BPR: {losses[1].item():.4f}, Sem: {losses[2].item():.4f})")
                else:
                    print(f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f}")
        
        # Финальная синхронизация в конце эпохи
        final_loss = total_loss_tensor.item()
        final_bpr = total_bpr_tensor.item()
        final_sem = total_sem_tensor.item()
        
        if isinstance(losses, tuple):
            print(f"🏁 [Epoch {epoch_idx}] Завершена! Суммарный лосс: {final_loss:.4f} (BPR: {final_bpr:.4f}, Sem: {final_sem:.4f})")
            return (final_loss, final_bpr, final_sem)
        else:
            print(f"🏁 [Epoch {epoch_idx}] Завершена! Суммарный лосс: {final_loss:.4f}")
            return final_loss

    def _check_nan(self, loss):
        # Переопределяем метод RecBole, чтобы отключить if torch.isnan(loss):
        # Который заставляет CPU ждать GPU на каждом батче!
        pass

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
        self.item_mapping_path = _cfg(config, 'item_mapping_path', 'clean_movies/clean_movies.item')
        self.ema_momentum = _cfg(config, 'ema_momentum', 0.99)
        self.hit_quantile = _cfg(config, 'hit_quantile', 0.80)
        self.semantic_sampling = _cfg(config, 'semantic_sampling', 'uniform')
        self.alpha_strategy = _cfg(config, 'alpha_strategy', 'dynamic_leaky_log')
        # New clearer names, with backward compatibility for old alpha_* keys.
        self.semantic_weight_cap = _cfg(config, 'semantic_weight_cap', _cfg(config, 'alpha_max', 0.5))
        self.semantic_weight_floor = _cfg(config, 'semantic_weight_floor', _cfg(config, 'alpha_min', 0.05))
        self.semantic_margin = _cfg(config, 'semantic_margin', 0.8)
        self.dynamic_cutoff_quantile = _cfg(config, 'dynamic_cutoff_quantile', 0.85)


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
                ).to(self.device)
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

            # --- Инициализация EMA буфера маяков (Dynamic Centroids) ---
            # Инициализируем маяки (EMA буфер) случайным шумом линейного слоя. 
            # Но теперь они будут обновляться только тогда, когда в батче есть популярные фильмы
            with torch.no_grad():
                if self.centroid_proj is not None:
                    init_dyn = self.centroid_proj(self.semantic_centroids)
                else:
                    init_dyn = self.semantic_centroids.clone()
            self.register_buffer('dynamic_centroids', init_dyn.clone())
            
        train_item_ids = dataset.inter_feat[dataset.iid_field]
        item_counts = torch.bincount(train_item_ids, minlength=self.n_items).float().to(self.device)
        self.item_counts = item_counts
        
        # Порог для "хитов". Фильмы с количеством оценок выше этого будут формировать центроид.
        active_items = item_counts[item_counts > 0]
        self.hit_threshold = torch.quantile(active_items, self.hit_quantile).item() if len(active_items) > 0 else 0

        # Все параметры расчета alpha-коэффициентов управляются через config.
        if self.alpha_strategy == 'dynamic_leaky_log':
            if len(active_items) > 0:
                dynamic_cutoff = torch.quantile(active_items, self.dynamic_cutoff_quantile).item()
                log_counts = torch.log(item_counts + 1)
                log_cutoff = torch.log(torch.tensor(dynamic_cutoff + 1, device=self.device))
                raw_alphas = 1.0 - (log_counts / log_cutoff)
            else:
                raw_alphas = torch.ones_like(item_counts)
            self.item_alpha_weights = torch.clamp(
                raw_alphas,
                min=self.semantic_weight_floor,
                max=self.semantic_weight_cap
            )
        else:
            raw_alphas = 1.0 / torch.log(torch.e + item_counts)
            self.item_alpha_weights = torch.clamp(raw_alphas, max=self.semantic_weight_cap)

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

        # --- ИЗМЕНЕНИЕ ДЛЯ СОВПАДЕНИЯ С INFERENCE (Борьба с Norm Bias) ---
        # Нормализуем векторы
        u_emb_norm = F.normalize(u_embeddings, p=2, dim=1)
        pos_emb_norm = F.normalize(pos_embeddings, p=2, dim=1)
        neg_emb_norm = F.normalize(neg_embeddings, p=2, dim=1)

        # Вычисляем косинусное сходство и масштабируем (Temperature Scaling)
        scale = 15.0
        pos_scores = torch.mul(u_emb_norm, pos_emb_norm).sum(dim=1) * scale
        neg_scores = torch.mul(u_emb_norm, neg_emb_norm).sum(dim=1) * scale
        
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
        # 1. Обновление маяков (EMA Update) идет по реальным взаимодействиям (pos_item) 
        # Это позволяет центроидам следовать за популярными фильмами графа
        cluster_ids = self.item2cluster[pos_item]
        
        if self.training:
            with torch.no_grad():
                # 1. Находим "хиты" и кастуем в float (Размер: Батч x 1)
                hit_mask_float = (self.item_counts[pos_item] >= self.hit_threshold).float().unsqueeze(1)
                
                # 2. Обнуляем эмбеддинги не-хитов (Размер: Батч x D)
                # Нет изменения размера тензора! Нет блокировки процессора!
                valid_embeddings = pos_embeddings * hit_mask_float
                
                # 3. Создаем One-Hot матрицу для всех элементов батча (Размер: Батч x K)
                cluster_one_hot = F.one_hot(cluster_ids, num_classes=self.n_clusters).float()
                
                # 4. Обнуляем One-Hot для не-хитов (Размер: Батч x K)
                cluster_one_hot_hits = cluster_one_hot * hit_mask_float
                
                # 5. Считаем сумму векторов для каждого кластера (Размер: K x D)
                sum_embeddings = torch.matmul(cluster_one_hot_hits.t(), valid_embeddings)
                
                # 6. Считаем количество хитов в каждом кластере (Размер: K x 1)
                counts = cluster_one_hot_hits.sum(dim=0).unsqueeze(1)
                
                # 7. Защита от деления на ноль (добавляем крошечный epsilon)
                c_k_tilde = sum_embeddings / (counts + 1e-9)
                
                # 8. Создаем маску обновления (1.0 для активных кластеров, 0.0 для пустых)
                update_mask = (counts > 0).float()
                
                # 9. Применяем EMA только там, где update_mask == 1.0 (Fallback для остальных)
                # Все операции выполняются одновременно для всей матрицы.
                # Используем .copy_(), чтобы гарантированно не отвязать registered_buffer.
                self.dynamic_centroids.copy_((
                    self.ema_momentum * self.dynamic_centroids + 
                    (1.0 - self.ema_momentum) * c_k_tilde
                ) * update_mask + self.dynamic_centroids * (1.0 - update_mask))
                        
        # 2. Нормализация динамических центроидов
        all_centroids_matrix = F.normalize(self.dynamic_centroids, dim=1)

        # 3. Отвязка сэмплирования для семантического лосса управляется через config.
        if self.semantic_sampling == 'uniform':
            batch_size = pos_item.shape[0]
            # n_items в RecBole включает отступ (padding) по индексу 0, поэтому сэмплируем от 1.
            semantic_items = torch.randint(1, self.n_items, (batch_size,), device=self.device)
        else:
            semantic_items = pos_item

        semantic_cluster_ids = self.item2cluster[semantic_items]
        semantic_item_embs = F.normalize(item_all_embeddings[semantic_items], dim=1)

        # Считаем лосс на равномерно сэмплированных фильмах
        cos_sim = torch.matmul(semantic_item_embs, all_centroids_matrix.t())
        
        # Получаем косинусное сходство элементов с их целевыми центроидами (маяками)
        batch_indices = torch.arange(batch_size, device=self.device)
        pos_sims = cos_sim[batch_indices, semantic_cluster_ids]
        
        # Создаем маску "зоны терпимости" (Dead-Zone Margin)
        # Если фильм уже достаточно близко к центру кластера (cos_sim > margin), отключаем для него семантический лосс, 
        # чтобы предотвратить коллапс ("идеальное среднее") и сохранить уникальность вектора.
        margin_mask = (pos_sims < self.semantic_margin).float()

        logits = cos_sim / self.tau
        proto_loss_vector = F.cross_entropy(logits, semantic_cluster_ids, reduction='none')

        # Per-item semantic weight + Margin Mask
        batch_alphas = self.item_alpha_weights[semantic_items]
        weighted_proto_loss_mean = (batch_alphas * proto_loss_vector * margin_mask).mean()
        
        sem_loss_value = self.cl_weight * weighted_proto_loss_mean

        # Возвращаем tuple, чтобы Trainer мог получить доступ к отдельным компонентам лосса
        return loss_bpr + sem_loss_value, loss_bpr, sem_loss_value

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        # L2-нормализация: убиваем влияние огромных векторов популярных фильмов!
        u_embeddings = F.normalize(u_embeddings, p=2, dim=1)
        i_embeddings = F.normalize(i_embeddings, p=2, dim=1)

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]

        # L2-нормализация для всей базы перед генерацией Топ-10
        u_embeddings = F.normalize(u_embeddings, p=2, dim=1)
        item_all_embeddings = F.normalize(item_all_embeddings, p=2, dim=1)

        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores.view(-1)