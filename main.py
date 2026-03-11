import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from adaptive_lightgcn import AdaptiveLightGCN, VerboseDataLoader

import glob, os
def clean_cache():
    for pth_file in glob.glob('clean_movies/*.pth') + glob.glob('*.pth'):
        try: os.remove(pth_file)
        except OSError: pass

_original_torch_load = torch.load

# Защита от двойного применения патча при перезапуске ячейки в Colab
if not hasattr(torch, '_original_load_saved'):
    torch._original_load_saved = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return torch._original_load_saved(*args, **kwargs)

    torch.load = patched_torch_load
# Подменяем оригинальную функцию нашей оберткой
torch.load = patched_torch_load

class DynamicLeakyLogAdaptiveLightGCN(AdaptiveLightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        train_item_ids = dataset.inter_feat[dataset.iid_field]
        item_counts = torch.bincount(train_item_ids, minlength=self.n_items).float().to(self.device)

        # --- 1. ДИНАМИЧЕСКИЙ ПОРОГ (85-й перцентиль) ---
        active_items = item_counts[item_counts > 0]
        dynamic_cutoff = torch.quantile(active_items, 0.85).item()

        # --- 2. LEAKY FLOOR (Минимальный вес) ---
        min_weight = 0.05 # Хиты сохранят 5% семантического веса (чтобы сохранить "мост")

        log_counts = torch.log(item_counts + 1)
        log_cutoff = torch.log(torch.tensor(dynamic_cutoff + 1).to(self.device))

        new_alphas = 1.0 - (log_counts / log_cutoff)

        # Обрезаем снизу не нулем, а min_weight!
        self.item_alpha_weights = torch.clamp(new_alphas, min=min_weight, max=1.0)

        tail_count = (item_counts <= 5).sum().item()
        head_count = (item_counts >= dynamic_cutoff).sum().item()

        print(f"\n[DynamicLeakyLog] 🌉 Применено Динамическое Дырявое Затухание!")
        print(f"   - Динамический порог (85%): {dynamic_cutoff:.1f} оценок")
        print(f"   - Min Weight (Floor): {min_weight} (сохраняем мост для {head_count} хитов)")
        print(f"   - Tail (<=5 оценок): {tail_count} фильмов (вес тяготеет к 1.0)")

# Очистка кэша
clean_cache()

# Конфигурация (K=150 возвращаем для базового сравнения)

print("\n🚀 Запуск Эксперимента 4: Dynamic Leaky Log Decay...")
config_dynamic = Config(model='LightGCN', dataset='clean_movies', config_file_list=['config.yaml'])

dataset_dynamic = create_dataset(config_dynamic)
train_dynamic, valid_dynamic, test_dynamic = data_preparation(config_dynamic, dataset_dynamic)

# Инициализация и обучение
model_dynamic = DynamicLeakyLogAdaptiveLightGCN(config_dynamic, train_dynamic.dataset).to(config_dynamic['device'])
trainer_dynamic = Trainer(config_dynamic, model_dynamic)

trainer_dynamic.fit(VerboseDataLoader(train_dynamic, interval=50), valid_dynamic, show_progress=False)

# Оценка
test_result_dynamic = trainer_dynamic.evaluate(test_dynamic, show_progress=False)
print("\n🏁 Глобальные метрики (Dynamic Leaky Log):")
print(test_result_dynamic)

print("\n📊 REPORT: Dynamic Leaky Log")
evaluate_stratified(trainer_dynamic, test_dynamic, train_dynamic)

with open('final_experiment_results.txt', 'a') as f:
    f.write(f"\n--- Dynamic Leaky Log Ablation ---\n")
    f.write(f"Recall@10: {test_result_dynamic['recall@10']}\n")
    f.write(f"NDCG@10: {test_result_dynamic['ndcg@10']}\n")