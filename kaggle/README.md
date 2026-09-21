# Обучение на GPU Kaggle

На CPU ноутбука полное обучение XLM-R + LoRA заняло бы 10–14 часов;
на T4 Kaggle — порядка 15 минут на прогон.

**Что уходит на Kaggle:** только синтетический корпус Қалқан (`corpus_train`,
`corpus_holdout`) и код. Внешние наборы на реальных звонках туда **не уходят** —
условия KorCCVi запрещают передачу данных. Обученный адаптер (несколько МБ)
скачивается обратно и оценивается на реальных звонках локально.

## Сборка

```bash
python kaggle/make_bundle.py                   # → kaggle/build/{qalqan-corpus,kernel}
python kaggle/make_bundle.py --seeds 42 43 44  # три сида — для устойчивости выводов
```

## Вариант А — через API (быстрее)

1. kaggle.com → Settings → API → **Create New Token** → `kaggle.json`
   в `C:\Users\<вы>\.kaggle\`. Аккаунт должен быть подтверждён по телефону
   (иначе ноутбуку не дадут GPU и Internet).
2. `pip install kaggle`
3. `python kaggle/make_bundle.py --push` — создаст private-датасет
   `qalqan-corpus` и запустит ноутбук `qalqan-train`.
4. Когда ноутбук отработает:
   `kaggle kernels output <user>/qalqan-train -p kaggle/output`
   → распаковать `qalqan_models.zip` в `models/`.

## Вариант Б — вручную

1. Kaggle → Datasets → **New Dataset** → перетащить все файлы из
   `kaggle/build/qalqan-corpus/` (кроме `dataset-metadata.json`), название
   `qalqan-corpus`, видимость **Private**.
2. Kaggle → Code → **New Notebook** → File → Import Notebook →
   `kaggle/build/kernel/qalqan_train.ipynb`.
3. Справа: Add Input → свой датасет `qalqan-corpus`; Settings →
   Accelerator **GPU T4 x2** (или P100), Internet **On**.
4. **Run All**. По окончании скачать из Output `qalqan_models.zip`
   и `qalqan_results.zip`, распаковать `qalqan_models.zip` в `models/`.

## Оценка на реальных звонках (локально)

```bash
cd src
python eval_detector.py --model-dir ../models/xlmr-lora-s42
```
