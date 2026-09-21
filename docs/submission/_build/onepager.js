// Қалқан — one-pager (RU + EN), A4, одна страница каждый.
// Он же служит «резюме проекта» (Положение, п. 8: не более одной страницы, RU и EN).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, BorderStyle, LevelFormat,
} = require("docx");

const OUT = process.argv[2] || ".";
const BLUE = "1C5CAB", INK = "0B0B0B", INK2 = "52514E", RULE = "C9C8C2", TINT = "EEF4FC";
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 850;              // A4, поля 1.5 см
const CONTENT_W = PAGE_W - 2 * MARGIN;

const run = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 21, bold: o.bold,
  italics: o.italics, color: o.color || INK });

function title(t, sub) {
  return [
    new Paragraph({ spacing: { after: 20 }, children: [run(t, { size: 34, bold: true, color: BLUE })] }),
    new Paragraph({ spacing: { after: 90 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 4 } },
      children: [run(sub, { size: 20, color: INK2 })] }),
  ];
}
function h(t) {
  return new Paragraph({ spacing: { before: 90, after: 30 },
    children: [run(t.toUpperCase(), { size: 20, bold: true, color: BLUE })] });
}
function p(parts, after = 40) {
  return new Paragraph({ spacing: { after, line: 252 },
    children: parts.map(x => typeof x === "string" ? run(x) : run(x[0], x[1])) });
}
function bullet(parts) {
  return new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 20, line: 247 },
    children: parts.map(x => typeof x === "string" ? run(x) : run(x[0], x[1])) });
}
function metrics(rows, head) {
  const w = [Math.round(CONTENT_W * 0.52), Math.round(CONTENT_W * 0.24)];
  w.push(CONTENT_W - w[0] - w[1]);
  const cell = (text, i, isHead) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: TINT } : undefined,
    margins: { top: 30, bottom: 30, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [run(text, { size: 19, bold: isHead || i === 1, color: i === 1 && !isHead ? BLUE : INK })] })],
  });
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
    borders: { top: border, bottom: border, left: border, right: border,
      insideHorizontal: border, insideVertical: border },
    rows: [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, i, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))],
  });
}

const RU = {
  file: "Qalqan_one-pager_RU.docx",
  title: "Қалқан — щит от телефонного мошенничества",
  sub: "Детектор, который предупреждает во время звонка — до того, как человек продиктует код. " +
       "Казахский, русский и смешанная речь. Трек AI for Finance.",
  author: "Автор: [ФИО], [курс, образовательная программа], SDU University · Репозиторий: github.com/daurenoralbek/qalqan",
  blocks: [
    ["Проблема", [
      p(["По данным Антифрод-центра Национального Банка РК (на 01.01.2026) — ",
        ["80 871 инцидент", { bold: true }], " с признаками мошенничества, из них ",
        ["22% — телефонное мошенничество", { bold: true }], ", 19% — фейковые инвестиции. Действующий контур ",
        ["реактивный", { bold: true }], ": инцидент разбирают после того, как человек уже продиктовал код из SMS, " +
        "перевёл деньги на «безопасный счёт» или установил программу удалённого доступа. Схемы обновляются " +
        "ежемесячно (2026: SMS-бомбинг + «Антиспам», «курьер» с кодом, «номер медицинской декларации», " +
        "вербовка дропперов, клонирование голоса), а правила на ключевых словах за ними не успевают."]),
    ]],
    ["Целевая аудитория", [
      p(["Банки РК (антифрод-подразделения) и операторы связи — как заказчики; Антифрод-центр НБ РК и АБРК — " +
        "как координаторы обмена моделью; клиенты банков, особенно пожилые и казахоязычные, — как те, " +
        "кого система защищает во время звонка."]),
    ]],
    ["Решение", [
      bullet([["Инкрементальная детекция. ", { bold: true }], "Речь переводится в текст (ASR), и после каждой реплики " +
        "модель XLM-RoBERTa + LoRA оценивает риск по всему разговору. Тревога звучит на стадии давления — " +
        "до запроса кода, перевода или установки программы."]),
      bullet([["Объяснимость. ", { bold: true }], "Рядом с риском — понятная причина: «требуют не класть трубку», " +
        "«просят код из SMS» (23 красных флага таксономии)."]),
      bullet([["Данные без персональных данных. ", { bold: true }], "Открытый синтетический корпус: 22 сценария в 11 темах, " +
        "каждый мошеннический сценарий привязан к публикации ведомства РК 2026 г. (НБ РК, Генпрокуратура, МВД, ЕНПФ, Минздрав, АФМ)."]),
      bullet([["Федеративное обучение. ", { bold: true }], "Банки обучают общую модель, не передавая записи разговоров: " +
        "обмен — только весами LoRA (≈6 МБ за раунд)."]),
    ]],
    ["Ценность — измерено", [
      metrics([
        ["Реальные звонки (вишинг; скам против звонков в банк; перевод на казахский), ROC AUC", "0.71–0.75", "0.50 — правила"],
        ["Тревога ДО целевого действия (при равном FPR 10–23%)", "91–94%", "37–58% — правила"],
        ["Каждый банк видел не все схемы: общая модель (FedAvg), val AUC", "0.839", "0.56–0.79 — банк в одиночку"],
      ], ["Показатель", "Қалқан", "Сравнение"]),
      p([["Честно: ", { bold: true }], "на самом трудном наборе — легитимный звонок на ту же тему, что и мошеннический, — " +
        "AUC пока 0.60; это главное направление доработки (расширение фразобанка)."], 20),
      p([["Главный эффект: ", { bold: true }], "предупреждение успевает прозвучать, пока деньги ещё у клиента. " +
        "Каждый предотвращённый инцидент — это не возврат через Антифрод-центр, а отсутствие ущерба."], 20),
    ]],
    ["Новизна", [
      p(["По нашим данным, первый в Казахстане разговорный детектор мошенничества с ", ["ранним предупреждением на казахском", { bold: true }],
        " и смешанной речи; открытый корпус с метрикой «тревога до целевого действия»; метод поиска артефакта " +
        "«тема звонка выдаёт класс» в синтетических данных (найден и устранён в ходе работы); федеративная схема " +
        "с LoRA для банков РК. Готовность — TRL 4 (прототип проверен на синтетике и на реальных публичных записях); " +
        "следующий шаг — пилот с банком на обезличенных звонках колл-центра."]),
    ]],
  ],
};

const EN = {
  file: "Qalqan_one-pager_EN.docx",
  title: "Qalqan — a shield against phone scams",
  sub: "A detector that warns during the call — before the victim reads out the code. " +
       "Kazakh, Russian and code-switched speech. Track: AI for Finance.",
  author: "Author: [Full name], [year, programme], SDU University · Repository: github.com/daurenoralbek/qalqan",
  blocks: [
    ["Problem", [
      p(["The Anti-Fraud Centre of the National Bank of Kazakhstan recorded ", ["80,871 fraud-related incidents", { bold: true }],
        " (as of 1 Jan 2026); ", ["22% are phone scams", { bold: true }], ", 19% fake investments. The current process is ",
        ["reactive", { bold: true }], ": incidents are handled after the person has already disclosed an SMS code, moved " +
        "money to a “safe account” or installed a remote-access app. Scripts change monthly (2026: SMS bombing + “Antispam”, " +
        "fake couriers asking for a code, the “medical declaration number”, money-mule recruitment, voice cloning), " +
        "and keyword rules cannot keep up."]),
    ]],
    ["Target users", [
      p(["Kazakhstani banks (anti-fraud units) and telecom operators as customers; the National Bank Anti-Fraud Centre and " +
        "the Association of Banks as coordinators of model sharing; bank clients — especially elderly and Kazakh-speaking " +
        "people — as those protected during the call."]),
    ]],
    ["Solution", [
      bullet([["Incremental detection. ", { bold: true }], "Speech is transcribed (ASR) and after every turn an " +
        "XLM-RoBERTa + LoRA model scores the risk of the whole conversation so far. The alert fires at the pressure " +
        "stage — before the request for a code, a transfer or an app install."]),
      bullet([["Explainable. ", { bold: true }], "Each alert comes with a plain reason: “they insist you stay on the line”, " +
        "“they ask for the SMS code” (23 red flags in the taxonomy)."]),
      bullet([["No personal data. ", { bold: true }], "Open synthetic corpus: 22 scenarios in 11 topics; every scam scenario " +
        "is tied to a 2026 public warning by a Kazakhstani authority (National Bank, Prosecutor General, Police, " +
        "Pension Fund, Ministry of Health, Financial Monitoring Agency)."]),
      bullet([["Federated learning. ", { bold: true }], "Banks train one shared model without exchanging call recordings — " +
        "only LoRA weights (≈6 MB per round) are shared."]),
    ]],
    ["Value — measured", [
      metrics([
        ["Real calls (vishing; scam vs genuine bank calls; Kazakh translation), ROC AUC", "0.71–0.75", "0.50 — rules"],
        ["Alert BEFORE the target action (at equal FPR 10–23%)", "91–94%", "37–58% — rules"],
        ["Each bank saw only some schemes: shared model (FedAvg), val AUC", "0.839", "0.56–0.79 — bank alone"],
      ], ["Metric", "Qalqan", "Baseline"]),
      p([["Honestly: ", { bold: true }], "on the hardest set — a genuine call on the same topic as the scam — " +
        "AUC is 0.60 so far; this is the main direction of further work (a larger phrasebank)."], 20),
      p([["Key effect: ", { bold: true }], "the warning arrives while the money is still with the client. " +
        "Every prevented incident means no loss at all rather than a recovery through the Anti-Fraud Centre."], 20),
    ]],
    ["Novelty", [
      p(["To our knowledge, the first conversational scam detector in Kazakhstan with ", ["early warning in Kazakh", { bold: true }],
        " and code-switched speech; an open corpus with an “alert before the target action” metric; a method to detect the " +
        "“call topic gives away the label” artefact in synthetic data (found and fixed during the project); federated LoRA " +
        "training for Kazakhstani banks. Readiness — TRL 4 (validated on synthetic data and real public recordings); " +
        "next step — a pilot with a bank on anonymised call-centre calls."]),
    ]],
  ],
};

async function build(L) {
  const children = [...title(L.title, L.sub)];
  for (const [head, body] of L.blocks) {
    children.push(h(head), ...body);
  }
  children.push(new Paragraph({ spacing: { before: 90 },
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 4 } },
    children: [run(L.author, { size: 17, color: INK2 })] }));
  const doc = new Document({
    creator: "Qalqan", title: L.title,
    styles: { default: { document: { run: { font: FONT, size: 21 } } } },
    numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 300, hanging: 220 } } } }] }] },
    sections: [{ properties: { page: { size: { width: PAGE_W, height: 16838 },
      margin: { top: 720, bottom: 640, left: MARGIN, right: MARGIN } } }, children }],
  });
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(path.join(OUT, L.file), buf);
  console.log("written", L.file);
}

(async () => { await build(RU); await build(EN); })();
