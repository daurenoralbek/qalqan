// Қалқан — техническое приложение (DOCX, A4). Разделы — по Положению, п. 8:
// методика/модель, датасет/исходники (ссылка на репозиторий), метрики/тест-кейс, план доработки (TRL).
// Числа федеративного эксперимента на реальных звонках читаются из results/federated (если уже посчитаны).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, LevelFormat, HeadingLevel, TableOfContents, PageNumber, Footer,
  ExternalHyperlink,
} = require("docx");

const OUT = process.argv[2];
const ROOT = process.argv[3];
const BLUE = "1C5CAB", INK = "0B0B0B", INK2 = "52514E", RULE = "C9C8C2", TINT = "EEF4FC";
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 1134, CW = PAGE_W - 2 * MARGIN;

const r = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 21, bold: o.bold, italics: o.italics,
  color: o.color || INK });
const P = (parts, o = {}) => new Paragraph({ spacing: { after: o.after ?? 100, line: 264 }, alignment: o.align,
  children: (Array.isArray(parts) ? parts : [parts]).map(x => typeof x === "string" ? r(x) : r(x[0], x[1])) });
const B = (parts) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 50, line: 264 },
  children: (Array.isArray(parts) ? parts : [parts]).map(x => typeof x === "string" ? r(x) : r(x[0], x[1])) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, keepLines: true, spacing: { before: 280, after: 120 },
  children: [new TextRun({ text: t, font: FONT, size: 30, bold: true, color: BLUE })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, keepLines: true, spacing: { before: 180, after: 80 },
  children: [new TextRun({ text: t, font: FONT, size: 24, bold: true, color: INK })] });
const CODE = (t) => new Paragraph({ spacing: { after: 20 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "F4F4F2" },
  children: [new TextRun({ text: t, font: "Consolas", size: 18, color: INK })] });

function table(head, rows, widths, o = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const w = widths.map(x => Math.round(CW * x / tot));
  w[w.length - 1] = CW - w.slice(0, -1).reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const cell = (t, i, isHead, bold) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: TINT } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [r(String(t), { size: 18, bold: isHead || bold })] })],
  });
  return [new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: w,
    borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
    rows: [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, i, true)) }),
      ...rows.map(row => new TableRow({ children: row.map((t, i) => cell(t, i, false, o.boldFirst && i === 0)) }))],
  }), new Paragraph({ spacing: { after: 120 }, children: [] })];
}

// ── федеративный эксперимент на реальных звонках (если посчитан) ─────────
function fedExt() {
  const dir = path.join(ROOT, "results", "federated");
  const names = { central: "Все данные в одном месте", fedavg: "FedAvg (обмен весами)",
    local0: "Банк 1 в одиночку", local1: "Банк 2 в одиночку", local2: "Банк 3 в одиночку" };
  const rows = [];
  for (const k of Object.keys(names)) {
    const f = path.join(dir, `detector_eval_ext_fed-topic-s42-${k}.json`);
    if (!fs.existsSync(f)) return null;
    const j = JSON.parse(fs.readFileSync(f, "utf-8")).sets;
    const g = (s) => j[s] && j[s].model.roc_auc != null ? j[s].model.roc_auc.toFixed(3) : "—";
    rows.push([names[k], g("korccvi_ru"), g("en_bank_ru"), g("en_topic_ru"), g("kazllm_kk")]);
  }
  return rows;
}

function build() {
  const c = [];
  // титул
  c.push(new Paragraph({ spacing: { after: 60 }, children: [r("Қалқан", { size: 52, bold: true, color: BLUE })] }));
  c.push(P([["Техническое приложение к заявке", { size: 30, bold: true }]], { after: 60 }));
  c.push(P([["Инкрементальный детектор мошеннических звонков на казахском, русском и смешанной речи", { size: 24, color: INK2 }]]));
  c.push(P([["Автор: ", { bold: true }], "[ФИО], [курс, образовательная программа], SDU University · Трек AI for Finance · 2026"]));
  c.push(new Paragraph({ spacing: { after: 100 }, children: [r("Репозиторий: ", { bold: true }),
    new ExternalHyperlink({ link: "https://github.com/daurenoralbek/qalqan",
      children: [new TextRun({ text: "github.com/daurenoralbek/qalqan", font: FONT, size: 21, color: BLUE, underline: {} })] })] }));
  c.push(P([["Готовность: ", { bold: true }], "TRL 4 — прототип проверен на синтетическом корпусе и на реальных публичных записях звонков."]));
  c.push(new TableOfContents("Содержание", { hyperlink: true, headingStyleRange: "1-2" }));

  // 1
  c.push(H1("1. Задача и метрики"));
  c.push(P("Система оценивает риск мошенничества по ходу разговора — после каждой реплики — и должна поднять тревогу " +
    "до того, как жертва выполнит целевое действие: продиктует код из SMS, переведёт деньги на «безопасный счёт», " +
    "установит программу удалённого доступа. Постановка следует работам по инкрементальной детекции " +
    "(arXiv:2609.20223) и crime-script подходу к стадиям мошеннического разговора (ScriptMind, arXiv:2601.13581)."));
  c.push(P("Метрики — на уровне диалога, одинаковые для модели и для правил:"));
  c.push(B([["Precision / Recall / F1 / FPR ", { bold: true }], "— тревога в диалоге считается срабатыванием; FPR — доля легитимных звонков с ложной тревогой."]));
  c.push(B([["ROC AUC ", { bold: true }], "по максимальному накопленному риску за диалог — качество ранжирования без выбора порога."]));
  c.push(B([["Доля тревог до целевого действия ", { bold: true }], "— главная метрика ценности: среди обнаруженных мошеннических диалогов с запросом целевого действия доля тех, где тревога прозвучала раньше этого запроса. Сравнивается с правилами при одинаковом FPR."]));
  c.push(B([["Медиана запаса ", { bold: true }], "— на сколько реплик тревога опередила запрос целевого действия."]));

  // 2
  c.push(H1("2. Данные"));
  c.push(H2("2.1. Синтетический корпус Қалқан (v2)"));
  c.push(P("Реальные записи звонков клиентов опубликовать нельзя, поэтому корпус порождается из документированной таксономии " +
    "(методика TeleAntiFraud-28k, arXiv:2503.24115). Каждая мошенническая схема привязана к публикации ведомства РК 2026 года " +
    "(НБ РК, Генпрокуратура, МВД, ЕНПФ, Минздрав, АФМ, полиция); персональных данных в корпусе нет."));
  c.push(...table(["Параметр", "Значение"], [
    ["Диалогов", "3 400 train + 600 val + 1 000 holdout (разбиение по шаблонам)"],
    ["Реплик", "≈ 32 600 на 4 000 диалогов"],
    ["Сценарии", "11 мошеннических + 11 легитимных = 11 тематических пар"],
    ["Языки", "русский 45%, казахский 25%, смешанная речь 30% (по диалогам)"],
    ["Разметка", "стадия S0–S6/SB, 20 детектируемых красных флагов, язык, момент целевого действия"],
    ["Профили жертвы", "доверчивый, сомневающийся, подозрительный, пожилой — управляют обрывом разговора"],
  ], [1, 3], { boldFirst: true }));
  c.push(H2("2.2. Меры против фиктивности задачи"));
  c.push(B([["Тема не выдаёт класс. ", { bold: true }], "Диалог порождается «тема → класс → сценарий»; приветствие и подтверждение личности (S0–S1) — из общего пула темы для обоих классов. Проверка: логистическая регрессия на символьных n-граммах только первой реплики даёт на holdout AUC 0.501 (в версии v1 — 0.911)."]));
  c.push(B([["Флаги в обоих классах. ", { bold: true }], "12 из 20 срабатывающих флагов встречаются и в легитимных звонках («не кладите трубку» у настоящей техподдержки); 28,8% легитимных диалогов содержат флаг, 29,7% мошеннических не содержат ни одного."]));
  c.push(B([["Длина не выдаёт класс. ", { bold: true }], "38% мошеннических диалогов обрываются жертвой до целевого действия; медиана длины обоих классов — 8 реплик."]));
  c.push(B([["Разбиение по шаблонам на три части. ", { bold: true }], "Holdout собран из формулировок, которых не было в обучении (пересечение русских реплик звонящего на стадиях манипуляции — 0,8%)."]));
  c.push(P([["Найденный и устранённый артефакт (v1 → v2). ", { bold: true }], "Модель, обученная на первой версии корпуса, почти всё решала по приветствию: тема звонка и стиль приветствия выдавали класс. На реальных звонках это проявилось как «звонок про банк = мошенничество» — AUC 0.412 на наборе «скам против звонков в банк». Перестройка корпуса подняла его до 0.733. Автоматические проверки темы и «начала разговора» добавлены в check_separability.py."]));
  c.push(H2("2.3. Внешний тест на реальных звонках"));
  c.push(P("Используется только для оценки, в обучении не участвует; сырые данные и переводы в репозиторий не входят (условия источников)."));
  c.push(...table(["Набор", "Содержание", "Язык"], [
    ["korccvi_ru", "200 реальных вишинг-звонков, опубликованных финрегулятором Кореи (FSS), против 200 обычных звонков (AI Hub)", "ko → ru (NLLB-200)"],
    ["en_bank_ru", "211 звонков скамеров со скам-бейтерами (YouTube) против 211 звонков в банковскую поддержку (HarperValleyBank)", "en → ru"],
    ["en_topic_ru", "98 тех же скам-звонков против 98 легитимных звонков на ту же тему", "en → ru"],
    ["kazllm_kk", "40 + 40 звонков из первых двух наборов, первые 15 реплик", "→ kk (ISSAI KazLLM-8B)"],
    ["youtube_kz", "4 публичные записи звонков мошенников жителям Казахстана", "ru (Whisper large-v3)"],
  ], [1.1, 4, 1.4], { boldFirst: true }));
  c.push(P("Публичных записей мошеннических звонков на казахском языке не найдено — это зафиксировано как ограничение; казахский тест — перевод реальных звонков."));

  // 3
  c.push(H1("3. Модель"));
  c.push(B([["Архитектура: ", { bold: true }], "xlm-roberta-base (280 млн параметров) + LoRA r = 16 на query/key/value и голова классификатора — 1,48 млн обучаемых параметров (0,53%)."]));
  c.push(B([["Вход: ", { bold: true }], "после реплики t — весь префикс разговора (реплики 0…t, разделённые [SEP], обрезка слева до 256 токенов). Роли говорящих не подаются — в ASR реальных звонков их нет."]));
  c.push(B([["Обучение: ", { bold: true }], "каждый префикс — пример с меткой диалога (≈ 27 500 префиксов); AdamW, lr 2·10⁻⁴, batch 32, 3 эпохи, fp16, Kaggle T4; выбор эпохи и калибровка — на val."]));
  c.push(B([["Накопитель риска: ", { bold: true }], "raw / EWMA / затухающий максимум поверх p_t; тип, параметр и порог выбираются на val (максимум тревог до целевого действия при FPR ≤ 5%)."]));
  c.push(B([["Бейзлайн: ", { bold: true }], "регулярные выражения по 20 красным флагам (ru/kk) с noisy-OR и затухающим максимумом — то, что банк может собрать за день."]));
  c.push(B([["Объяснение: ", { bold: true }], "реплика с наибольшим ростом риска и сработавшие красные флаги (redflags.explain)."]));

  // 4
  c.push(H1("4. Результаты"));
  c.push(H2("4.1. Реальные звонки (ROC AUC, среднее ± разброс по 3 сидам)"));
  c.push(...table(["Набор", "Қалқан", "Правила"], [
    ["Реальный вишинг, Корея → ru", "0.751 ± 0.032", "0.505"],
    ["Скам против звонков в банк, en → ru", "0.733 ± 0.052", "0.505"],
    ["Скам против легитимных на ту же тему, en → ru", "0.598 ± 0.026", "0.505"],
    ["Перевод на казахский (KazLLM)", "0.709 ± 0.057", "0.512"],
    ["Звонки мошенников казахстанцам (n = 4)", "—", "0 из 4"],
  ], [3, 1.3, 1], { boldFirst: true }));
  c.push(P("Правила на реальной речи — на уровне случайного угадывания: мошеннический смысл в текстах есть («прокуратура/полиция» — 56% мошеннических против 8% легитимных звонков), но сформулирован иначе, чем в регулярных выражениях. Порог, подобранный на синтетической валидации, на реальные звонки не переносится (FPR 0.40–0.74) — в развёртывании его калибрует банк на своих данных."));
  c.push(H2("4.2. Шаблонный holdout при одинаковом FPR"));
  c.push(...table(["FPR", "Recall правил", "Recall модели", "До действия: правила", "До действия: модель"], [
    ["5%", "0.388", "0.320", "37%", "89%"],
    ["10%", "0.388", "0.392", "37%", "94%"],
    ["23%", "0.590", "0.538", "58%", "91%"],
  ], [1, 1.2, 1.2, 1.4, 1.4]));
  c.push(P("ROC AUC на holdout: модель 0.674 ± 0.003, правила 0.724. Holdout не отложен для правил: регулярные выражения писались по всему фразобанку, включая формулировки holdout; для модели эти формулировки новые. По первой реплике модель класс не определяет (AUC 0.48–0.51)."));
  c.push(H2("4.3. Федеративное обучение"));
  c.push(P("Три «банка» получают непересекающиеся части обучающего корпуса; равный вычислительный бюджет (3 прохода по данным): централизованно — 3 эпохи; каждый банк в одиночку — 3 эпохи на своих данных; FedAvg — 3 раунда по 1 локальной эпохе, взвешенное усреднение весов LoRA и головы. Обмен — 5,9 МБ за обновление (106 МБ на весь эксперимент); записи разговоров банк не покидают."));
  c.push(...table(["Разбиение (val ROC AUC)", "Централизованно", "FedAvg", "Банки в одиночку"], [
    ["Случайное (IID)", "0.837", "0.876", "0.839 / 0.846 / 0.862"],
    ["По профилю клиента", "0.874", "0.860", "0.857 / 0.850 / 0.830"],
    ["По темам (банк видел только часть схем)", "0.858", "0.839", "0.786 / 0.556 / 0.746"],
  ], [2.6, 1.3, 1, 2], { boldFirst: true }));
  const fe = fedExt();
  if (fe) {
    c.push(P([["Разбиение «по темам» на реальных звонках (ROC AUC):", { bold: true }]], { after: 60 }));
    c.push(...table(["Модель", "Вишинг", "Скам vs банк", "Та же тема", "Казахский"], fe, [2.4, 1, 1.1, 1, 1], { boldFirst: true }));
  }
  c.push(P("Когда каждый банк видит только часть схем, модель одного банка плохо переносится на чужие схемы (val AUC до 0.556), а FedAvg почти догоняет общий датасет (0.839 против 0.858) без передачи записей. При случайном и «профильном» разбиении все варианты близки — каждый банк и так видит все темы."));

  // 5
  c.push(H1("5. Тест-кейсы"));
  c.push(P("Четыре диалога из шаблонного holdout (формулировки не встречались при обучении) встроены в демо (demo/examples.json) и воспроизводятся ссылкой вида ?example=N&play=all:"));
  c.push(...table(["#", "Диалог", "Ожидаемое поведение (проверено)"], [
    ["1", "Мошенник-«техподдержка», рус.: «сбой в приложении» → «установите QuickSupport»", "Модель: тревога с реплики 4 — за 3 реплики до запроса; правила: только на реплике 7"],
    ["2", "Настоящая техподдержка по заявке клиента, каз.: «Телефонды қоймаңыз»", "Модель: риск 0%, тревоги нет; правила: ложная тревога на «не кладите трубку»"],
    ["3", "«Курьер» просит код из SMS, каз.", "Модель: тревога на стадии давления (реплики 3–5); правила: на просьбе кода (реплика 7)"],
    ["4", "Настоящее уведомление оператора о замене SIM, каз./рус.", "Модель: 0%; правила: ложная тревога на «кодтарды ешкімге айтпаңыз»"],
  ], [0.3, 3, 3.4]));
  c.push(P("Функциональная проверка демо автоматизирована через streamlit.testing (AppTest): проигрывание примеров, тревога, подпись о запасе реплик."));

  // 6
  c.push(H1("6. Ограничения"));
  for (const t of [
    "Корпус синтетический: нет перебиваний, оговорок, шума; реплики жертвы подбираются по стадии, а не по смыслу предыдущей фразы.",
    "Фразобанк мал, особенно казахский: у многих пулов 2–3 варианта, поэтому val частично пересекается с обучением по шаблонам (порог с val оптимистичен).",
    "Казахский текст не вычитан носителем; казахские паттерны правил беднее русских.",
    "Внешний тест — перевод (NLLB, KazLLM) и ASR; живой казахской речи мошенников в открытом доступе нет.",
    "ASR для казахского искажает ключевые слова (Whisper: «ешкімге айтпаңыз» → «еш кім ғайтпаңыз»).",
    "Клонированный голос (SC08) по тексту почти не отличить от настоящей просьбы родственника — нужен акустический модуль.",
  ]) c.push(B(t));

  // 7
  c.push(H1("7. План доработки (TRL)"));
  c.push(...table(["Уровень", "Срок", "Что делаем"], [
    ["TRL 4 — сейчас", "сентябрь 2026", "Прототип, корпус v2, внешний тест, федеративный эксперимент, демо"],
    ["TRL 5", "Q4 2026 – Q1 2027", "Пилот с банком на обезличенных звонках колл-центра; калибровка порога; вычитка и расширение казахского фразобанка носителями"],
    ["TRL 6", "Q2 2027", "Потоковый ASR kk/ru (модели ISSAI); API для антифрода банка или оператора; задержка < 1 с на реплику"],
    ["TRL 7", "H2 2027", "Пилот с клиентами через приложение банка или сеть оператора, с согласия клиента; акустический модуль против клонирования голоса"],
    ["Сеть", "2028", "Федерация банков через АБРК / Антифрод-центр НБ РК: общая модель без обмена записями"],
  ], [1.2, 1.4, 4.2], { boldFirst: true }));
  c.push(P([["Ресурсы. ", { bold: true }], "Прототип — без затрат на вычисления (Kaggle T4, ~2 GPU-часа; открытые модели XLM-R, NLLB-200, Whisper, ISSAI KazLLM). Пилот (оценка): 1 GPU-сервер для ASR и модели — порядка $400–600 в месяц в облаке; вычитка казахского фразобанка носителем — около 40 часов; интеграция с антифродом — силами банка-партнёра."]));

  // 8
  c.push(H1("8. Воспроизведение"));
  c.push(P("Структура репозитория: data/taxonomy (таксономия), data/generated (корпус), src (генерация, проверки, обучение, оценка), demo (Streamlit), kaggle (пакет для GPU), docs (карточка датасета, документы заявки), results (метрики)."));
  for (const l of [
    "cd src",
    "python generate_corpus.py && python make_template_holdout.py",
    "python validate_corpus.py && python check_separability.py",
    "python run_baseline.py --corpus ../data/generated/corpus_holdout.jsonl --split holdout",
    "python ../kaggle/make_bundle.py --seeds 42 43 44 --push      # обучение на Kaggle GPU",
    "python eval_detector.py --model-dir ../models/xlmr-lora-s42   # оценка локально",
    "python ../kaggle/make_bundle.py --job federated --push        # федеративный эксперимент",
    "streamlit run ../demo/app.py                                  # демо",
  ]) c.push(CODE(l));
  c.push(P("Подробности, лицензии внешних данных и полный отчёт о корпусе — docs/dataset_card.md и data/external/README.md в репозитории.", { after: 60 }));

  return new Document({
    creator: "Qalqan", title: "Қалқан — техническое приложение", features: { updateFields: true },
    styles: { default: { document: { run: { font: FONT, size: 21 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 30, bold: true, color: BLUE }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 24, bold: true, color: INK }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
      ] },
    numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1000, bottom: 1000, left: MARGIN, right: MARGIN } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [r("Қалқан — техническое приложение · стр. ", { size: 16, color: INK2 }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: INK2 })] })] }) },
      children: c,
    }],
  });
}
(async () => {
  const buf = await Packer.toBuffer(build());
  fs.writeFileSync(OUT, buf);
  console.log("written", OUT, fe_status());
})();

function fe_status() { return fedExt() ? "(с федеративными метриками на реальных звонках)" : "(без федеративных метрик на реальных звонках — ещё не посчитаны)"; }
