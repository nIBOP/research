import torch
import random
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from adaptive_lightgcn import AdaptiveLightGCN, CustomTrainer
import glob, os, sys, datetime
import numpy as np
import pandas as pd
from recbole.utils.case_study import full_sort_topk

class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Создаем уникальный файл лога для каждого запуска
os.makedirs("train_logs", exist_ok=True)
log_filename = os.path.join("train_logs", datetime.datetime.now().strftime("experiment_log_%Y-%m-%d_%H-%M-%S.txt"))
sys.stdout = OutputLogger(log_filename)
sys.stderr = sys.stdout  # Перенаправляем и ошибки в этот же файл

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def evaluate_stratified(trainer, test_data, train_data, k=10):
    print("\n" + "="*50)
    print("📊 СТРАТИФИЦИРОВАННЫЙ АНАЛИЗ (HEAD / TORSO / TAIL)")

    model = trainer.model
    device = model.device

    # 1. Извлекаем и считаем популярность всех фильмов из обучающего графа
    # Используем iid_field из train_data.dataset
    dataset = train_data.dataset
    iid_field = dataset.iid_field
    # ВАЖНО: берем inter_feat именно из train_data.dataset, чтобы считать популярность по трейну
    train_item_ids = dataset.inter_feat[iid_field].numpy()
    item_counts = pd.Series(train_item_ids).value_counts()
    total_items = len(item_counts)

    # 2. Задаем границы корзин (20% / 30% / 50%)
    head_cutoff = int(total_items * 0.20)
    torso_cutoff = int(total_items * 0.50) # 20% + 30%

    # 3. Разбиваем на множества (set) для быстрого поиска
    head_items = set(item_counts.index[:head_cutoff])
    torso_items = set(item_counts.index[head_cutoff:torso_cutoff])
    tail_items = set(item_counts.index[torso_cutoff:])

    print(f"🎬 Распределение базы ({total_items} фильмов):")
    print(f"   - Head (Топ 20%): {len(head_items)} элементов")
    print(f"   - Torso (Средние 30%): {len(torso_items)} элементов")
    print(f"   - Tail (Редкие 50%): {len(tail_items)} элементов")

    # 4. Оценка
    hits = {'head': 0, 'torso': 0, 'tail': 0}
    total_gt = {'head': 0, 'torso': 0, 'tail': 0}

    # Семплируем пользователей для ускорения
    uid_field = dataset.uid_field
    test_user_ids = test_data.dataset.inter_feat[uid_field].unique().numpy()

    target_users = test_user_ids
    print(f"\n🔍 Оценка на выборке из {len(target_users)} пользователей...")

    for uid in target_users:
        # Получаем Ground Truth для пользователя
        user_mask = (test_data.dataset.inter_feat[uid_field].numpy() == uid)
        user_test_items = test_data.dataset.inter_feat[iid_field].numpy()[user_mask]

        if len(user_test_items) == 0: continue

        # Получаем предсказания модели
        _, topk_tensor = full_sort_topk(torch.tensor([uid]), model, test_data, k=k, device=device)
        topk_items = topk_tensor[0].cpu().numpy()

        for item in user_test_items:
            # Определяем категорию item
            category = 'tail'
            if item in head_items: category = 'head'
            elif item in torso_items: category = 'torso'

            total_gt[category] += 1

            if item in topk_items:
                hits[category] += 1

    # 5. Вывод результатов
    print("\n📈 Результаты Recall@10:")
    for category in ['head', 'torso', 'tail']:
        recall = hits[category] / total_gt[category] if total_gt[category] > 0 else 0.0
        print(f"   - {category.capitalize()}: {recall:.4f} (Hits: {hits[category]}/{total_gt[category]})")

    print("="*50 + "\n")

def clean_cache():
    for pth_file in glob.glob('clean_movies/*.pth') + glob.glob('*.pth'):
        try: os.remove(pth_file)
        except OSError: pass

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Подменяем оригинальную функцию нашей оберткой
torch.load = patched_torch_load

# Очистка кэша
clean_cache()

# Конфигурация (K=150 возвращаем для базового сравнения)

print("\n🚀 Запуск Эксперимента 4: Dynamic Leaky Log Decay...")
config_dynamic = Config(model='LightGCN', dataset='clean_movies', config_file_list=['config.yaml'])

dataset_dynamic = create_dataset(config_dynamic)
train_dynamic, valid_dynamic, test_dynamic = data_preparation(config_dynamic, dataset_dynamic)

# Инициализация и обучение
model_dynamic = AdaptiveLightGCN(config_dynamic, train_dynamic.dataset).to(config_dynamic['device'])

# Используем наш CustomTrainer вместо обычной обертки
trainer_dynamic = CustomTrainer(config_dynamic, model_dynamic)

# show_progress=False отключает стандартный tqdm, но наш _train_epoch будет печатать логи
trainer_dynamic.fit(train_dynamic, valid_dynamic, show_progress=False)

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