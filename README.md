# Валидация понятности жестов

Этот репозиторий нужен для валидации результатов разметки жестов на понятность.

Для получения результатов работы предполагается использовать cli:

```bash
python ./src/cli/validate_markup.py путь-до-файла-с-разметкой путь-до-словаря-схлопываний путь-до-расширенного-словаря-схлопываний минимальное-значение-для-близости
```

Например,

```bash
python ./src/cli/validate_markup.py ./res_marks_sberai_p1.tsv ./data/raw/clap_rules/clap_rules.json ./data/raw/clap_rules/clap_rules_extended.json 0.7
```

Здесь расширенный словарь схлопываний - это словарь, в который добавили потенциально приемлимые, но еще не подтвержденные схлопывания, например, течение \<-> длина

На выходе будет создана таблица качества с accuracy по каждому из классов, а также с усредненным значением accuracy. Также для каждого случая будут отрисованы графики с точностью.

Предполагается, что этот код будет использоваться для более объективного оценивания результатов работы различных версий модели. Страничка с результатами экспериментов и актуальными версиями словаря живет [здесь](https://www.notion.so/maximazzik/5ded383160a043b1881a84e8d31adfa8).
