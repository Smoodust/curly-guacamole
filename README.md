# AIIJC: EMCAD experiments

Код находится в `src/`, настройки экспериментов — в `configs/`, результаты — в `runs/<name>/`. Окружение Python: `challenges`.

Базовый pipeline этой ветки — **PVT-v2-B2 + EMCAD + forensic-ветка**, вход 640. Его не нужно собирать цепочкой флагов:

- mixed full-frame/crop обучение: по умолчанию 50% полных кадров, 50% кропов, foreground-guided выбор половины кропов; последние две эпохи full-frame;
- DCT с правильным natural-порядком таблицы Pillow, без legacy-переключателя;
- фиксированный train/development/holdout manifest и проверенные originals с нулевой маской в train;
- валидация по исходным маскам: восстановление вероятностей до оригинального размера перед порогом;
- выбор лучшего checkpoint и порогов только по development AIC, отдельные срезы provided/originals.

## Эксперименты

| Конфиг | Отличие от baseline |
|---|---|
| [baseline](configs/baseline.yaml) | Базовая модель, BCE + Dice на всех примерах |
| [dct576](configs/dct576.yaml) | Baseline с текущей forensic-веткой, RGB 576 |
| [rgb576](configs/rgb576.yaml) | Та же модель без forensic-ветки, RGB 576 |
| [jpeg576](configs/jpeg576.yaml) | JPEG Artifact Module и компактная пирамида вместо текущей ветки, RGB 576; [подробности](docs/jpeg_ablation.md) |
| [jpeg576_pretrained](configs/jpeg576_pretrained.yaml) | Тот же `jpeg576`, JPEG Artifact Module стартует с весов `DCT_djpeg.pth` |
| [positive_dice](configs/positive_dice.yaml) | Dice только на позитивных; BCE по всем |
| [positive_dice_full_train](configs/positive_dice_full_train.yaml) | 4 sampled-эпохи + 1 полный full-frame проход; warmup, постоянный LR, затем cosine до 10% |
| [local](configs/local.yaml) | Positive Dice + локальная RGB/residual-ветка 1024 |
| [luma](configs/luma.yaml) | Positive Dice + компактная яркостная ветка 1024 на GPU; [устройство и проверки](docs/luma_branch.md) |
| [stride4](configs/stride4.yaml) | Spatial refinement 144 каналов после EMCAD |
| [stride2_rgb](configs/stride2_rgb.yaml) | RGB refinement stride 2, batch 2 × accumulation 8 |
| [dct_aux](configs/dct_aux.yaml) | Дополнительная DCT-голова с весом 0.2 |

Общая база — 6 эпох × 24 000 показов, batch 4 × accumulation 4, seed 42, EMA, BF16. `local`, `luma` и `positive_dice_full_train` наследуют `positive_dice`; остальные варианты — `baseline`. Гиперпараметры можно менять, общие исправления отключить нельзя.

Все параметры явно перечислены в `configs/baseline.yaml`. Остальные рецепты наследуют его и задают только отличия. Загрузчик также поддерживает минимальный конфиг со встроенными defaults:

```yaml
paths:
  run_name: baseline
```

Все значения по умолчанию определены в [src/config.py](src/config.py) и [AugmentationConfig](src/data/augmentation/base.py). Snapshots сохраняют полностью разрешённые настройки и `pipeline_version: emcad_v1`.

## Запуск

`positive_dice_full_train` наследует positive_dice, но использует 5 эпох: четыре по 24 000
показов с возвращением и одну по всему train без повторов (87 728 записей в текущем
протоколе, включая originals). Всего 183 728 показов; финальный баланс классов естественный.
Последний неполный батч сохраняется. Аугментации прежние, в финале отключены кропы.
`train.full_train_epochs: 1` включает этот режим и двухфазный LR: warmup на 5% шагов
основной фазы (0.2 эпохи), постоянный базовый LR до её конца, затем cosine за полный
проход до `min_lr_factor: 0.1`. Число шагов учитывает длину каждой фазы и accumulation.
Запуск с нуля; для resume того же прогона нельзя менять параметры расписания или batch.

Откройте [notebooks/train.ipynb](notebooks/train.ipynb), выберите `experiment` и выполните ячейки. До обучения ноутбук проверяет manifest и полный FLOPs-бюджет без загрузки pretrained-весов. Последняя ячейка запускает обучение.

Из корня проекта:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.training --config configs/local.yaml
```

Новый эксперимент содержит только отличия:

```yaml
extends: positive_dice.yaml
paths:
  run_name: longer
train:
  epochs: 18
```

`extends` разрешается относительно YAML; словари объединяются, списки заменяются. `.env`/переменные среды `AIIJC_DATA_PATH` и `AIIJC_RUNS_PATH` задают машинные пути. `dataset.protocol_path` — необязательная настройка расположения обязательного manifest; по умолчанию используется сохранённый `runs/validation_protocol_20260908/protocol`, даже если текущая рабочая папка другая. Это не переключатель старой/новой валидации.

Параметры оборудования можно переопределить в `.env` или переменных процесса:

```dotenv
AIIJC_BATCH_SIZE=4
AIIJC_ACCUM_STEPS=4
AIIJC_AMP=bf16
AIIJC_DEVICE=cuda
AIIJC_DEVICES=auto
AIIJC_DISTRIBUTED_BACKEND=auto
AIIJC_WORKERS=10
```

Приоритет: переменные процесса → `.env` → YAML, включая наследуемые рецепты. Пустое значение оставляет настройку из YAML. Для `amp` допустимы `off`, `fp16`, `bf16`. Пример есть в [.env.example](.env.example). Переопределения применяются при `load_experiment_config`; snapshots сохраняют фактические значения, а `ExperimentConfig.from_dict(snapshot)` не подменяет параметры обучения настройками текущей видеокарты. Batch size, accumulation и AMP могут влиять на результат обучения; одинаковый effective batch не гарантирует одинаковое поведение BatchNorm.

По умолчанию `resume: false`; занятое имя получает суффикс. Для продолжения укажите фактическое имя папки и `train.resume: true`. Версия pipeline, модель, loss, augmentation, геометрия и provenance проверяются до перезаписи snapshot.

### Обучение на нескольких GPU

Обычный запуск `python -m src.training --config configs/baseline.yaml` и
`ExperimentRunner(config).run()` в Python/ноутбуке автоматически используют все
видимые CUDA-карты. При одной карте обучение остаётся однопроцессным. При нескольких
создаётся по процессу на карту; отдельный `torchrun` не нужен и не поддерживается.

```yaml
train:
  device: cuda
  devices: auto             # Все видимые GPU; либо список, например [0, 1].
  distributed_backend: auto # NCCL/DDP при наличии, иначе Gloo с обменом через CPU.
  batch_size: 4             # На одну GPU.
  accum_steps: 4
  workers: 4                # На один процесс/GPU.
```

Для одной выбранной карты задайте `device: cuda:1` с `devices: auto` либо
`device: cuda` с `devices: [1]`. Индексы относятся к картам, видимым процессу с учётом
`CUDA_VISIBLE_DEVICES`. Для CPU/MPS используется одно устройство. Отсутствующая
CUDA-карта или недоступный явно выбранный backend вызывает ошибку до обучения.
В `.env` список задаётся как `AIIJC_DEVICES="[0, 1]"`. Переменная `AIIJC_DEVICE=cuda:0`
при `devices: auto` ограничивает обучение одной картой.

Эффективный полный батч равен `batch_size × accum_steps × число GPU`: с настройками
выше и двумя картами — 32 изображения. Learning rate автоматически не изменяется.
BatchNorm остаётся локальным для каждой карты. Градиенты нормируются по общему
числу примеров в накоплении; conditional loss, включая positive Dice, вычисляется
локально, поэтому совпадение с обучением одним большим батчем не гарантируется.

`epoch_size` задаёт общее число выборок. Weighted sampling сначала создаёт единый
поток, затем делит его между процессами. При обычном обучении неполный глобальный
батч отбрасывается. При включённом `full_train_epochs` данные не дополняются
дубликатами: размеры локальных батчей балансируются, каждый пример финального
прохода используется ровно один раз. Если данных слишком мало для одинакового
числа непустых батчей, нужно уменьшить число GPU или увеличить `batch_size`.

Валидация автоматически распределяется между теми же GPU: каждой достаётся
непересекающаяся последовательная часть целых батчей, без повторов и пропусков.
Границы батчей сохраняются, чтобы не менять группировку примеров при расчёте loss.
После прохода главный процесс собирает компактные гистограммы в исходном порядке,
подбирает общие пороги и сохраняет OOF. Loss агрегируется по суммам и числу примеров;
AIC считается по объединённым данным, а не усреднением оценок отдельных GPU.
Прогресс показывает локальные батчи главного процесса. Пустые части набора допустимы;
ошибка валидации одного процесса передаётся остальным. Логи и checkpoint пишет
главный процесс. Дополнительных настроек или изменения `batch_size` не требуется.
EMA и checkpoint сохраняются без префикса
`module.`, поэтому подходят для существующего инференса. Для contrastive-вариантов
checkpoint хранит RNG каждого процесса. `resume` требует прежнее число GPU;
смена числа карт выполняется через новый запуск с `finetune_from`.

При NCCL используется PyTorch DDP. При Gloo вычисления остаются на GPU, а градиенты
передаются через CPU блоками около 25 MiB и усредняются перед шагом оптимизатора.
Это медленнее NCCL, но обходит отсутствие CUDA all-reduce в Windows-сборках Gloo.
Неиспользуемые параметры сохраняют `grad=None`, если они не участвовали в loss
ни на одной карте. Buffer'ы модели синхронизируются перед forward, EMA — перед
валидацией и checkpoint. Подробности ограничения Windows:
[issue PyTorch](https://github.com/pytorch/pytorch/issues/186535).

## Оценка и результаты

### Профиль скорости обучения

Изолированный CPU-микробенчмарк local-препроцессинга, без датасета и GPU:

```bash
python -m src.data.benchmark_local --output profiles/local_preprocess.json
```

Сравнивает прежний полный float32-буфер с записью групп по три канала сразу в итоговый AMP-буфер, проверяет побитовое совпадение и измеряет цветовое преобразование, нормализацию, разности, resize, cast и запись CHW. Использует синтетические исходники 640×640, 1024×1024 и 1536×2048, один CPU-поток, чередующийся порядок замеров. Можно задать `--dtype float16` или `--dtype float32`. В worker-профиле преобразование local-типа теперь включено в `local_features`; отдельный `local_cast` равен нулю. Сравнивайте сумму этих двух этапов с прежним отчётом.

Для замера подготовки изображений внутри DataLoader-воркеров:

```bash
AIIJC_WORKERS=4 python -m src.training.profiling --config configs/local.yaml --profile-data --batches 64 --output profiles/data.json
```

Блок `results.loader.worker_profile` содержит среднее, p50, p95 (мс на изображение) и долю каждого этапа: read_image, read_mask, qtable, setup, jpeg, dct, geometry, photometric, local_features, local_cast, resize, tensorize. Учитываются только потреблённые батчи после warmup. Это wall time внутри воркеров, включая возможные паузы планировщика; сборка батча, IPC, pinning и очереди сюда не входят. Время параллельных воркеров нельзя складывать с временем CUDA. Тайминги возвращаются в основной процесс вместе с батчем и удаляются до отправки на GPU. В обычном обучении эта диагностика выключена.

Local-входы для CUDA передаются в точности выбранного AMP: BF16 или FP16 (30 MiB вместо 60 MiB на изображение). Разности и resize вычисляются в float32; при `amp=off` и на CPU формат остаётся float32. Обучение использует отдельный copy stream с ожиданием готовности данных на compute stream, чтобы перенос следующего батча мог перекрываться с уже поставленными в очередь вычислениями. Состояние модели и формат checkpoint не меняются.

Сравнить прежнюю передачу float32, только компактную передачу и компактную передачу с отдельным stream:

```bash
python -m src.training.profiling --config configs/local.yaml --compare-transfer --batches 64 --output profiles/transfer.json
```

В итоговой строке `Transport comparison` сравнивайте wall time режимов `legacy`, `compact`, `optimized`. Для `optimized` поле `transfer` показывает неперекрытое ожидание compute stream, а не полную длительность копирования. Порядок этого короткого сравнения фиксирован; небольшие различия требуют повторного замера.

На сервере с CUDA, отдельно от работающего обучения:

```bash
python -m src.training.profiling --config configs/local.yaml --warmup 8 --batches 32 --output profiles/local.json
```

Команда использует параметры оборудования из `.env`. Две временные модели без загрузки pretrained и без сохранения запусков сравнивают обычный DataLoader (`loader`) с повторением одного готового батча на GPU (`gpu_replay`). Warmup исключён из статистики. Отчёт содержит wall time на батч и CUDA Events для transfer, forward+loss, backward, optimizer (включая clipping), EMA+scheduler и учёта метрик. Для optimizer/EMA также выводится время на обновление, учитывающее accumulation. CUDA Events измеряют интервалы в stream, включая паузы подачи команд CPU; это не сумма длительностей отдельных kernels. Replay исключает загрузку/перенос данных, но использует один фиксированный батч, поэтому является диагностическим сравнением. Архитектура и loss берутся из выбранного конфига; существующие веса и результаты не изменяются.

При сборке submission путь к данным берётся из `AIIJC_DATA_PATH` текущей среды (переменная процесса → `.env` → `global_config.DATA_PATH`), а не из snapshot обучения. Относительный путь разрешается от корня проекта. Явный аргумент `data_path` / `--data-path` имеет приоритет; без отдельного `template_path` шаблон submission также читается из выбранной папки данных.

- [Протокол валидации](docs/validation_protocol.md): правила development и однократного holdout.
- [Карта экспериментов](docs/experiment_map.md): соответствие старых рецептов новым и архив.
- [Локальная ветка](docs/emcad_local_branch.md): архитектура, geometry и вычислительный бюджет.
- [notebooks/experiments.ipynb](notebooks/experiments.ipynb): отдельные таблицы текущих запусков и архива.

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.inference runs/local submissions/local
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.eval --run runs/local
```

Вторая команда — окончательный holdout, запускается только после выбора кандидата. Протокол берётся из snapshot; holdout не участвует в подборе порогов.

Выполненные EMCAD-запуски сохранены под короткими именами конфигов в `runs/` (ранние baseline имеют суффиксы `legacy` и `legacy_dct`), остальные исторические веса, метрики и карточки — в `runs/archive/`. Их конфиги описывают предыдущий pipeline: для воспроизведения используйте код `main`. Новая ветка явно отклоняет старые checkpoints без версии, чтобы не подменять их DCT или архитектуру.

## Проверки

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m pytest
```

Пределы соревнования: 100 GFLOPs и 50 мс/изображение на H100. FLOPs считаются полным eval-forward через `FlopCounterMode`. Измерение H100 latency с preprocessing и улучшение качества требуют серверного эксперимента; рефакторинг сам обучение не запускает.
