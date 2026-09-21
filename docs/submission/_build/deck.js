// Қалқан — pitch-deck (12 слайдов, 16:9). Разделы — по Положению конкурса, п. 8:
// рынок/контекст, архитектура/метод, данные, метрики, план внедрения, риски, бюджет/ресурсы.
const pptxgen = require("pptxgenjs");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const OUT = process.argv[2] || "Qalqan_pitch-deck.pptx";

const C = {
  navy: "14213D", navy2: "1F3157", sky: "2A78D6", orange: "EB6834", gold: "F2B705",
  red: "D03B3B", green: "0CA30C", ink: "1F2937", muted: "5B6475", line: "E3E6EC",
  tint: "F2F5FA", white: "FFFFFF", skyTint: "E8F1FC", orangeTint: "FDEEE7", redTint: "FBEAEA",
};
const F = "Calibri";

async function icon(Comp, color, size = 256) {
  const svg = renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

function T(slide, text, o) {                    // текстовый блок с общими настройками
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: F, color: C.ink, margin: 0,
    valign: "top" }, o));
}
function title(slide, text, sub) {
  T(slide, text, { x: 0.6, y: 0.4, w: 12.1, h: 0.7, fontSize: 32, bold: true, color: C.navy });
  if (sub) T(slide, sub, { x: 0.6, y: 1.1, w: 12.1, h: 0.45, fontSize: 16, color: C.muted });
}
function card(slide, x, y, w, h, fill = C.tint) {
  slide.addShape("roundRect", { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function footer(slide, text) {
  T(slide, text, { x: 0.6, y: 7.0, w: 12.1, h: 0.3, fontSize: 10, color: C.muted });
}
async function iconCircle(slide, Comp, x, y, d, bg, fg) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg } });
  const pad = d * 0.24;
  slide.addImage({ data: await icon(Comp, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";                   // 13.33 × 7.5 in
  pres.title = "Қалқан — детектор мошеннических звонков";
  pres.author = "[ФИО], SDU University";

  // ── 1. Титул ─────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    s.addImage({ data: await icon(fa.FaShieldAlt, C.gold), x: 0.9, y: 1.35, w: 1.25, h: 1.25 });
    T(s, "Қалқан", { x: 2.45, y: 1.2, w: 9, h: 1.4, fontSize: 72, bold: true, color: C.white });
    T(s, "Детектор мошеннических звонков, который предупреждает до того, как вы продиктуете код",
      { x: 0.9, y: 3.0, w: 11.2, h: 1.1, fontSize: 28, color: C.white });
    T(s, "Казахский · русский · смешанная речь   |   XLM-RoBERTa + LoRA   |   федеративное обучение",
      { x: 0.9, y: 4.25, w: 11.5, h: 0.5, fontSize: 18, color: "BFD4F2" });
    T(s, "[ФИО], [курс, образовательная программа] · SDU University\nТрек AI for Finance · Республиканский конкурс АБРК / КБТУ / NU, 2026",
      { x: 0.9, y: 5.9, w: 11.5, h: 0.9, fontSize: 14, color: "BFD4F2" });
    s.addNotes("Здравствуйте. Проект «Қалқан» — по-казахски «щит». Это система, которая слушает телефонный разговор " +
      "и предупреждает человека о мошенничестве во время звонка — до того, как он продиктует код из SMS или переведёт деньги.");
  }

  // ── 2. Проблема и контекст ───────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Мошенник успевает раньше, чем антифрод",
      "Антифрод-центр Национального Банка РК, данные на 01.01.2026");
    const stats = [["80 871", "инцидент с признаками\nмошенничества", C.navy],
                   ["22%", "телефонное\nмошенничество", C.red],
                   ["19%", "фейковые\nинвестиции", C.orange]];
    stats.forEach(([n, l, col], i) => {
      const x = 0.6 + i * 2.75;
      card(s, x, 1.85, 2.5, 2.3);
      T(s, n, { x: x + 0.25, y: 2.05, w: 2.1, h: 1.0, fontSize: 44, bold: true, color: col });
      T(s, l, { x: x + 0.25, y: 3.1, w: 2.1, h: 0.9, fontSize: 15, color: C.ink });
    });
    T(s, "Заблокировано 2,8 млрд ₸, возвращено пострадавшим > 500 млн ₸ — но это работа после инцидента.",
      { x: 0.6, y: 4.4, w: 8.1, h: 0.6, fontSize: 15, color: C.muted, italic: true });
    card(s, 9.05, 1.85, 3.65, 4.6, C.redTint);
    await iconCircle(s, fa.FaExclamationTriangle, 9.3, 2.05, 0.7, C.red, C.white);
    T(s, "Контур реактивный", { x: 10.15, y: 2.2, w: 2.5, h: 0.5, fontSize: 18, bold: true, color: C.red });
    T(s, [
      { text: "Инцидент разбирают, когда код уже продиктован, а деньги — на «безопасном счёте».", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "Схемы 2026 года:", options: { bold: true, breakLine: true } },
      { text: "SMS-бомбинг + «Антиспам»", options: { bullet: true, breakLine: true } },
      { text: "«курьер» просит код", options: { bullet: true, breakLine: true } },
      { text: "«номер медицинской декларации»", options: { bullet: true, breakLine: true } },
      { text: "вербовка дропперов", options: { bullet: true, breakLine: true } },
      { text: "клонирование голоса", options: { bullet: true } },
    ], { x: 9.3, y: 2.95, w: 3.2, h: 3.4, fontSize: 14, paraSpaceAfter: 3 });
    T(s, "Правила на ключевых словах не успевают за новыми сценариями — формулировки меняются каждый месяц.",
      { x: 0.6, y: 5.3, w: 8.1, h: 0.9, fontSize: 18, bold: true, color: C.navy });
    footer(s, "Источники: Антифрод-центр НБ РК; предупреждения Генпрокуратуры, МВД, Минздрава, АФМ РК, полиции (2026).");
    s.addNotes("По данным Антифрод-центра Нацбанка — более 80 тысяч инцидентов, из них 22% — телефонное мошенничество. " +
      "Действующий контур реактивный: он разбирает инцидент после того, как человек уже продиктовал код. " +
      "А схемы меняются каждый месяц — в 2026 году появились SMS-бомбинг с «Антиспамом», фальшивые курьеры, " +
      "«номер медицинской декларации», вербовка дропперов.");
  }

  // ── 3. Идея: предупредить на стадии давления ─────────────────────
  {
    const s = pres.addSlide();
    title(s, "Идея: предупредить во время звонка — на стадии давления",
      "Мошеннический звонок развивается по сценарию. Решение принимается ДО запроса целевого действия.");
    const st = [["S0", "Контакт"], ["S1", "Доверие"], ["S2", "Угроза или\nприманка"], ["S3", "Изоляция и\nдавление"],
                ["S4", "«Следователь»"], ["S5", "Код / перевод /\nAnyDesk"], ["S6", "Закрепление"]];
    const x0 = 0.6, w = 1.62, gap = 0.12, y = 3.1;
    st.forEach(([id, name], i) => {
      const x = x0 + i * (w + gap);
      const hot = i === 5;
      card(s, x, y, w, 1.35, hot ? C.redTint : C.tint);
      T(s, id, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.35, fontSize: 14, bold: true, color: hot ? C.red : C.muted });
      T(s, name, { x: x + 0.15, y: y + 0.48, w: w - 0.3, h: 0.8, fontSize: 14, bold: true, color: C.ink });
    });
    const mark = (i, text, col, row) => {
      const x = x0 + i * (w + gap);
      const yy = row === "top" ? 2.05 : 4.75;
      s.addShape("roundRect", { x: x - 0.05, y: yy, w: w + 0.1 + (i < 6 ? w : 0), h: 0.8,
        fill: { color: col }, line: { color: col }, rectRadius: 0.08 });
      T(s, text, { x: x + 0.1, y: yy + 0.08, w: (i < 6 ? 2 * w : w) - 0.1, h: 0.65, fontSize: 14, bold: true,
        color: C.white, valign: "middle" });
    };
    mark(2, "Қалқан: тревога здесь — в 91–94% случаев до кода", C.sky, "top");
    mark(5, "Правила: здесь — уже поздно", C.orange, "bottom");
    T(s, "Антифрод-центр сегодня: после S6 — когда деньги ушли", { x: 0.6, y: 5.9, w: 12, h: 0.5, fontSize: 16,
      color: C.muted, italic: true });
    footer(s, "Стадии — crime script по ScriptMind (arXiv:2601.13581); постановка инкрементальной детекции — arXiv:2609.20223.");
    s.addNotes("Любой мошеннический звонок — это сценарий: контакт, доверие, угроза, изоляция и давление, и только потом " +
      "просьба назвать код. Правила на ключевых словах срабатывают на самой просьбе — это уже поздно. Мы поднимаем тревогу " +
      "на стадии давления — в 91–94% случаев раньше, чем прозвучит просьба кода.");
  }

  // ── 4. Как работает ──────────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Как работает", "После каждой реплики модель пересчитывает риск по всему разговору");
    const steps = [[fa.FaPhoneAlt, "Звонок", "казахский, русский,\nсмешанная речь"],
                   [fa.FaMicrophone, "ASR", "речь → текст\n(ISSAI / Whisper)"],
                   [fa.FaBrain, "XLM-R + LoRA", "вероятность мошенничества\nпо всему сказанному"],
                   [fa.FaChartLine, "Накопитель риска", "сглаживание и порог,\nподобранные на данных"],
                   [fa.FaBell, "Тревога + причина", "«требуют не класть трубку»,\n«просят код из SMS»"]];
    const w = 2.25, gap = 0.2;
    for (let i = 0; i < steps.length; i++) {
      const [ic, h, d] = steps[i];
      const x = 0.6 + i * (w + gap);
      card(s, x, 1.9, w, 2.9);
      await iconCircle(s, ic, x + (w - 0.9) / 2, 2.1, 0.9, i === 4 ? C.red : C.sky, C.white);
      T(s, h, { x: x + 0.15, y: 3.15, w: w - 0.3, h: 0.5, fontSize: 16, bold: true, color: C.navy, align: "center" });
      T(s, d, { x: x + 0.15, y: 3.65, w: w - 0.3, h: 1.0, fontSize: 13, color: C.ink, align: "center" });
      if (i < steps.length - 1) {
        s.addShape("rightArrow", { x: x + w + 0.01, y: 3.2, w: gap - 0.02, h: 0.28, fill: { color: C.muted }, line: { color: C.muted } });
      }
    }
    const facts = [["1,48 млн", "обучаемых параметров из 280 млн (0,5%) — LoRA"],
                   ["без ролей", "не нужно знать, кто говорит: работает на сыром ASR"],
                   ["3 языка", "одна модель для kk, ru и смешанной речи"]];
    facts.forEach(([n, l], i) => {
      const x = 0.6 + i * 4.1;
      T(s, n, { x, y: 5.25, w: 3.8, h: 0.6, fontSize: 28, bold: true, color: C.sky });
      T(s, l, { x, y: 5.85, w: 3.8, h: 0.7, fontSize: 14, color: C.ink });
    });
    s.addNotes("Речь переводится в текст. После каждой реплики модель XLM-RoBERTa с LoRA оценивает вероятность " +
      "мошенничества по всему сказанному, накопитель сглаживает риск, и при превышении порога человек получает " +
      "предупреждение с понятной причиной. Обучается только полпроцента параметров — это важно для федеративного обучения.");
  }

  // ── 5. Данные ────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Данные и проверка на реальных звонках",
      "Обучение — на синтетике без персональных данных; оценка — на реальных публичных записях");
    const big = [["22", "сценария"], ["11", "тем звонков"], ["~32 600", "реплик"], ["0", "персональных данных"]];
    big.forEach(([n, l], i) => {
      const x = 0.6 + i * 3.05;
      T(s, n, { x, y: 1.7, w: 2.9, h: 0.8, fontSize: 40, bold: true, color: C.navy });
      T(s, l, { x, y: 2.45, w: 2.9, h: 0.4, fontSize: 15, color: C.muted });
    });
    card(s, 0.6, 3.0, 5.95, 3.55);
    await iconCircle(s, fa.FaDatabase, 0.85, 3.2, 0.65, C.navy, C.white);
    T(s, "Синтетический корпус Қалқан", { x: 1.65, y: 3.3, w: 4.8, h: 0.5, fontSize: 18, bold: true, color: C.navy });
    T(s, [
      { text: "11 мошеннических схем — каждая из публикации ведомства РК 2026: НБ РК, Генпрокуратура, МВД, ЕНПФ, Минздрав, АФМ", options: { bullet: true, breakLine: true } },
      { text: "11 легитимных звонков на те же темы: настоящий банк, курьер, техподдержка", options: { bullet: true, breakLine: true } },
      { text: "Разметка по стадиям и 23 красным флагам", options: { bullet: true, breakLine: true } },
      { text: "Отложенный тест — на формулировках, которых модель не видела", options: { bullet: true } },
    ], { x: 0.85, y: 4.0, w: 5.5, h: 2.5, fontSize: 14, paraSpaceAfter: 4 });
    card(s, 6.75, 3.0, 5.95, 3.55, C.skyTint);
    await iconCircle(s, fa.FaGlobeAsia, 7.0, 3.2, 0.65, C.sky, C.white);
    T(s, "Внешний тест на реальных звонках", { x: 7.8, y: 3.3, w: 4.8, h: 0.5, fontSize: 18, bold: true, color: C.navy });
    T(s, [
      { text: "200 реальных вишинг-звонков, опубликованных финрегулятором Кореи (FSS), против 200 обычных", options: { bullet: true, breakLine: true } },
      { text: "211 звонков скамеров против 211 звонков в банк (публичные записи)", options: { bullet: true, breakLine: true } },
      { text: "Перевод на казахский — модель ISSAI KazLLM (партнёр конкурса)", options: { bullet: true, breakLine: true } },
      { text: "4 записи звонков мошенников жителям Казахстана", options: { bullet: true } },
    ], { x: 7.0, y: 4.0, w: 5.5, h: 2.5, fontSize: 14, paraSpaceAfter: 4 });
    footer(s, "Реальных казахоязычных записей мошеннических звонков в открытом доступе нет — это зафиксировано как ограничение.");
    s.addNotes("Реальные записи звонков клиентов публиковать нельзя, поэтому мы построили синтетический корпус по методике " +
      "TeleAntiFraud: 22 сценария в 11 темах, каждая мошенническая схема — из официального предупреждения. " +
      "Но синтетике нельзя верить на слово, поэтому модель проверена на реальных звонках — вишинге, записях скамеров " +
      "и их переводе на казахский моделью KazLLM от ISSAI.");
  }

  // ── 6. Честная синтетика ─────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Нашли и устранили ловушку синтетических данных",
      "Первая версия корпуса: тема звонка выдавала ответ, и модель выучила «звонок про банк = мошенничество»");
    const rows = [["AUC простой модели по одному приветствию", "0.911", "0.501"],
                  ["Риск по приветствию настоящего банка", "≈ 100%", "≈ 45% — «пока неясно»"],
                  ["Реальные звонки: скам против звонков в банк, AUC", "0.41", "0.73"]];
    card(s, 0.6, 1.9, 5.4, 4.6, C.redTint);
    card(s, 7.3, 1.9, 5.4, 4.6, C.skyTint);
    T(s, "Корпус v1", { x: 0.9, y: 2.05, w: 4.8, h: 0.5, fontSize: 20, bold: true, color: C.red });
    T(s, "Корпус v2", { x: 7.6, y: 2.05, w: 4.8, h: 0.5, fontSize: 20, bold: true, color: C.sky });
    rows.forEach(([lbl, a, b], i) => {
      const y = 2.75 + i * 1.2;
      T(s, a, { x: 0.9, y, w: 4.8, h: 0.6, fontSize: 30, bold: true, color: C.red });
      T(s, lbl, { x: 0.9, y: y + 0.6, w: 4.8, h: 0.5, fontSize: 13, color: C.ink });
      T(s, b, { x: 7.6, y, w: 4.8, h: 0.6, fontSize: 30, bold: true, color: C.sky });
      T(s, lbl, { x: 7.6, y: y + 0.6, w: 4.8, h: 0.5, fontSize: 13, color: C.ink });
    });
    s.addShape("rightArrow", { x: 6.2, y: 3.9, w: 0.9, h: 0.6, fill: { color: C.navy }, line: { color: C.navy } });
    T(s, "11 пар «мошенничество — легитимный звонок на ту же тему», общие приветствия, автопроверка в CI-скрипте",
      { x: 0.6, y: 6.6, w: 12.1, h: 0.4, fontSize: 14, bold: true, color: C.navy });
    s.addNotes("Важный момент честности. Первая модель показала на синтетике почти 0.98 — и провалилась на реальных звонках " +
      "в банк: она выучила, что звонок про банк — это мошенничество. Мы нашли причину — тема и стиль приветствия выдавали класс — " +
      "перестроили корпус и добавили автоматическую проверку. Теперь по приветствию класс не угадать, а на реальных звонках " +
      "в банк качество выросло с 0.41 до 0.73.");
  }

  // ── 7. Результаты ────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Результаты: работает на реальных звонках — правила нет",
      "ROC AUC на реальных звонках, среднее по 3 запускам (0.5 — случайное угадывание)");
    s.addChart(pres.charts.BAR, [
      { name: "Қалқан", labels: ["Вишинг\n(Корея → ru)", "Скам против\nзвонков в банк", "Скам против\nтой же темы", "Перевод на\nказахский"],
        values: [0.751, 0.733, 0.598, 0.709] },
      { name: "Правила", labels: ["Вишинг\n(Корея → ru)", "Скам против\nзвонков в банк", "Скам против\nтой же темы", "Перевод на\nказахский"],
        values: [0.505, 0.505, 0.505, 0.512] },
    ], { x: 0.5, y: 1.65, w: 7.6, h: 5.1, barDir: "col", barGapWidthPct: 60, chartColors: [C.sky, C.orange],
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
      showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelFontSize: 12,
      dataLabelColor: C.ink, showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: C.ink,
      catAxisLabelColor: C.ink, catAxisLabelFontSize: 12, valAxisLabelColor: C.muted, valAxisLabelFontSize: 11,
      valGridLine: { color: C.line, size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false,
      valAxisLineShow: false });
    card(s, 8.55, 1.75, 4.15, 2.35, C.skyTint);
    T(s, "91–94%", { x: 8.8, y: 1.9, w: 3.8, h: 0.9, fontSize: 44, bold: true, color: C.sky });
    T(s, "тревог — ДО запроса кода или перевода\n(правила: 37–58%) при одинаковой доле ложных тревог",
      { x: 8.8, y: 2.85, w: 3.8, h: 1.1, fontSize: 13, color: C.ink });
    card(s, 8.55, 4.3, 4.15, 2.35);
    T(s, "0 из 4", { x: 8.8, y: 4.45, w: 3.8, h: 0.9, fontSize: 44, bold: true, color: C.orange });
    T(s, "реальных звонков мошенников жителям Казахстана распознали правила: живая речь не совпадает с их ключевыми словами",
      { x: 8.8, y: 5.4, w: 3.8, h: 1.1, fontSize: 13, color: C.ink });
    footer(s, "Самый трудный набор — легитимный звонок на ту же тему (0.60): главное направление доработки. Порог нужно калибровать на данных банка.");
    s.addNotes("На реальных звонках модель, обученная только на синтетике, даёт AUC 0.71–0.75; правила — 0.5, то есть " +
      "угадывание: живая речь не совпадает с их ключевыми словами. При одинаковой доле ложных тревог модель предупреждает " +
      "до запроса кода в 91–94% случаев, правила — в 37–58%. Честно о слабом месте: легитимный звонок на ту же тему " +
      "различается хуже — 0.60, и порог для реальных звонков нужно калибровать на данных банка.");
  }

  // ── 8. Федеративное обучение ─────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Банки обучают модель вместе, не передавая разговоры",
      "Эксперимент: 3 банка, каждый видел только часть мошеннических схем. Метрика — ROC AUC на валидации");
    // схема: 3 банка → сервер
    for (let i = 0; i < 3; i++) {
      const y = 1.95 + i * 1.45;
      card(s, 0.6, y, 2.3, 1.15, C.tint);
      await iconCircle(s, fa.FaUniversity, 0.75, y + 0.2, 0.75, C.navy, C.white);
      T(s, `Банк ${i + 1}`, { x: 1.6, y: y + 0.25, w: 1.25, h: 0.35, fontSize: 15, bold: true, color: C.navy });
      T(s, "звонки — у себя", { x: 1.6, y: y + 0.6, w: 1.25, h: 0.35, fontSize: 11, color: C.muted });
      s.addShape("rightArrow", { x: 3.0, y: y + 0.42, w: 0.75, h: 0.3, fill: { color: C.sky }, line: { color: C.sky } });
    }
    card(s, 3.9, 2.7, 1.9, 2.35, C.skyTint);
    await iconCircle(s, fa.FaSyncAlt, 4.45, 2.9, 0.8, C.sky, C.white);
    T(s, "Усреднение\nвесов LoRA", { x: 4.0, y: 3.85, w: 1.7, h: 0.7, fontSize: 14, bold: true, color: C.navy, align: "center" });
    T(s, "5,9 МБ за раунд", { x: 4.0, y: 4.55, w: 1.7, h: 0.4, fontSize: 12, color: C.muted, align: "center" });
    s.addChart(pres.charts.BAR, [
      { name: "Общая модель", labels: ["Все данные\nв одном месте", "FedAvg\n(обмен весами)", "Банк 1\nв одиночку", "Банк 2\nв одиночку", "Банк 3\nв одиночку"],
        values: [0.858, 0.839, 0, 0, 0] },
      { name: "Банк в одиночку", labels: ["Все данные\nв одном месте", "FedAvg\n(обмен весами)", "Банк 1\nв одиночку", "Банк 2\nв одиночку", "Банк 3\nв одиночку"],
        values: [0, 0, 0.786, 0.556, 0.746] },
    ], { x: 6.2, y: 1.75, w: 6.5, h: 4.55, barDir: "col", barGrouping: "stacked", barGapWidthPct: 55,
      chartColors: [C.sky, "9AA3B2"], valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25,
      valAxisLabelFormatCode: "0.00", showValue: true, dataLabelFormatCode: "0.00;;;", dataLabelPosition: "inEnd",
      dataLabelColor: C.white, dataLabelFontSize: 13, dataLabelFontBold: true, showLegend: false,
      catAxisLabelColor: C.ink, catAxisLabelFontSize: 11, valAxisLabelColor: C.muted, valAxisLabelFontSize: 11,
      valGridLine: { color: C.line, size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false,
      valAxisLineShow: false });
    T(s, "Банк, который не встречал схему, её не ловит (до 0.56). Обмен весами почти догоняет общий датасет — без передачи записей.",
      { x: 0.6, y: 6.4, w: 12.1, h: 0.55, fontSize: 15, bold: true, color: C.navy });
    s.addNotes("Банки не могут обмениваться записями разговоров. Мы смоделировали три банка, каждый из которых видел только " +
      "часть схем. В одиночку банк плохо ловит схемы, которых у него не было — до 0.56. Федеративное усреднение весов LoRA " +
      "даёт 0.84 — почти как если бы все данные были в одном месте, — а передаётся всего 6 мегабайт весов за раунд.");
  }

  // ── 9. Демо ──────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Демо: риск растёт по ходу разговора",
      "Реплики из отложенного набора — модель их не видела. Streamlit-прототип, видео до 5 минут");
    const lab = Array.from({ length: 11 }, (_, i) => String(i + 1));
    s.addChart(pres.charts.LINE, [
      { name: "Қалқан", labels: lab, values: [0.45, 0.453, 0.446, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] },
      { name: "Правила", labels: lab, values: [0, 0, 0, 0, 0, 0, 0.93, 0.837, 0.9, 0.9, 0.81] },
    ], { x: 0.5, y: 1.95, w: 6.7, h: 4.2, chartColors: [C.sky, C.orange], lineSize: 2, lineDataSymbol: "circle",
      lineDataSymbolSize: 7, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%",
      showLegend: true, legendPos: "t", legendFontSize: 12, catAxisTitle: "реплика", showCatAxisTitle: true,
      catAxisTitleFontSize: 11, catAxisLabelColor: C.muted, valAxisLabelColor: C.muted, catAxisLabelFontSize: 11,
      valAxisLabelFontSize: 11, valGridLine: { color: C.line, size: 0.75 }, catGridLine: { style: "none" } });
    T(s, "Мошенник-«техподдержка» (рус.)", { x: 0.6, y: 1.65, w: 6.5, h: 0.35, fontSize: 15, bold: true, color: C.navy });
    const notes = [
      [C.sky, "Реплика 4 — тревога модели", "«Вижу ваш договор… В приложении банка обнаружен сбой» — за 3 реплики до просьбы"],
      [C.orange, "Реплика 7 — правила", "«Установите QuickSupport» — программа уже устанавливается"],
      [C.green, "Настоящая техподдержка (каз.)", "«Телефонды қоймаңыз, желіні тексеремін» — правила: ложная тревога; модель: 0%"],
    ];
    notes.forEach(([col, h, d], i) => {
      const y = 1.95 + i * 1.45;
      card(s, 7.6, y, 5.1, 1.25, C.tint);
      s.addShape("ellipse", { x: 7.8, y: y + 0.22, w: 0.3, h: 0.3, fill: { color: col }, line: { color: col } });
      T(s, h, { x: 8.3, y: y + 0.15, w: 4.25, h: 0.4, fontSize: 15, bold: true, color: C.navy });
      T(s, d, { x: 8.3, y: y + 0.55, w: 4.25, h: 0.65, fontSize: 12, color: C.ink });
    });
    T(s, "Видео-демо: [ссылка]   ·   Код: github.com/daurenoralbek/qalqan", { x: 0.6, y: 6.45, w: 12.1, h: 0.4,
      fontSize: 14, bold: true, color: C.navy });
    s.addNotes("Вот как это выглядит. Мошенник представляется техподдержкой. На четвёртой реплике — «вижу ваш договор, " +
      "в приложении сбой» — модель уже поднимает тревогу. Правила молчат до седьмой реплики, когда человек уже ставит " +
      "программу удалённого доступа. А на настоящей техподдержке на казахском правила дают ложную тревогу на фразе " +
      "«не кладите трубку», а модель понимает контекст и остаётся на нуле.");
  }

  // ── 10. План внедрения ───────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "План внедрения: от прототипа к пилоту", "Сейчас — TRL 4: прототип проверен на синтетике и реальных публичных записях");
    const steps = [["TRL 4", "Сейчас", "Прототип, корпус, внешний тест, федеративный эксперимент, демо", C.sky],
                   ["TRL 5", "Q4 2026 – Q1 2027", "Пилот с банком: обезличенные звонки колл-центра, калибровка порога, вычитка казахского носителями", C.navy],
                   ["TRL 6", "Q2 2027", "Потоковый ASR kk/ru (модели ISSAI), API для антифрода банка или оператора, задержка < 1 с", C.navy],
                   ["TRL 7", "H2 2027", "Пилот с клиентами: приложение банка или сеть оператора, с согласия клиента", C.navy],
                   ["Сеть", "2028", "Федерация банков через АБРК / Антифрод-центр НБ РК: общая модель без обмена записями", C.gold]];
    const w = 2.3, gap = 0.15;
    steps.forEach(([trl, when, what, col], i) => {
      const x = 0.6 + i * (w + gap);
      s.addShape("ellipse", { x: x + (w - 1.0) / 2, y: 1.95, w: 1.0, h: 1.0, fill: { color: col }, line: { color: col } });
      T(s, trl, { x: x + (w - 1.0) / 2, y: 1.95, w: 1.0, h: 1.0, fontSize: 15, bold: true,
        color: col === C.gold ? C.navy : C.white, align: "center", valign: "middle" });
      if (i < steps.length - 1) s.addShape("line", { x: x + (w + 1.0) / 2, y: 2.45, w: w + gap - 1.0, h: 0,
        line: { color: C.line, width: 2 } });
      T(s, when, { x, y: 3.1, w, h: 0.4, fontSize: 14, bold: true, color: C.navy, align: "center" });
      card(s, x, 3.6, w, 2.35, C.tint);
      T(s, what, { x: x + 0.15, y: 3.75, w: w - 0.3, h: 2.1, fontSize: 13, color: C.ink });
    });
    T(s, "Точки интеграции: антифрод банка (входящие на номера клиентов по согласию) · оператор связи · мобильное приложение банка",
      { x: 0.6, y: 6.25, w: 12.1, h: 0.5, fontSize: 14, bold: true, color: C.navy });
    s.addNotes("Сейчас проект на уровне TRL 4. Следующий шаг — пилот с банком на обезличенных звонках колл-центра: там " +
      "калибруется порог и вычитывается казахский. Затем интеграция с потоковым распознаванием речи от ISSAI и API для " +
      "антифрода, пилот с клиентами — и сеть банков через Ассоциацию банков и Антифрод-центр.");
  }

  // ── 11. Риски ────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    title(s, "Риски и как мы их снимаем");
    const rows = [
      ["Синтетика ≠ живая речь", "Внешний тест на реальных звонках; пилот на обезличенных звонках банка; расширение фразобанка с носителями"],
      ["Порог не переносится на реальные звонки", "Калибровка порога на данных каждого банка — часть федеративного протокола"],
      ["Ошибки ASR на казахском", "Модели ISSAI; обучение на тексте с ошибками распознавания"],
      ["Приватность и закон о персональных данных", "Анализ на стороне банка или устройства, обмен только весами, согласие клиента"],
      ["Мошенники меняют сценарии", "Федеративное дообучение на новых случаях; таксономия обновляется по сводкам ведомств"],
      ["Клонированный голос по тексту почти не отличить", "Акустический модуль (следующий этап): текст ограничен сверху"],
    ];
    const tbl = [[{ text: "Риск", options: { bold: true, color: C.white, fill: { color: C.navy } } },
                  { text: "Что делаем", options: { bold: true, color: C.white, fill: { color: C.navy } } }]];
    rows.forEach(([r, m], i) => tbl.push([
      { text: r, options: { bold: true, color: C.navy, fill: { color: i % 2 ? C.white : C.tint } } },
      { text: m, options: { color: C.ink, fill: { color: i % 2 ? C.white : C.tint } } }]));
    s.addTable(tbl, { x: 0.6, y: 1.45, w: 12.1, colW: [4.2, 7.9], fontFace: F, fontSize: 14, rowH: 0.72,
      valign: "middle", border: { type: "solid", pt: 0.75, color: C.line }, margin: [0.06, 0.12, 0.06, 0.12] });
    s.addNotes("Главные риски. Синтетика не равна живой речи — поэтому внешний тест и пилот. Порог не переносится — его " +
      "калибрует каждый банк. Приватность — анализ на стороне банка, наружу уходят только веса. Мошенники меняют сценарии — " +
      "федеративное дообучение. Клонированный голос по тексту почти не отличить — нужен акустический модуль.");
  }

  // ── 12. Бюджет, ресурсы, команда ─────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    T(s, "Ресурсы, команда и что нужно для пилота", { x: 0.6, y: 0.4, w: 12.1, h: 0.7, fontSize: 32, bold: true, color: C.white });
    const cols = [
      [fa.FaCoins, "Прототип", [
        "Вычисления: 0 ₸ — Kaggle T4, ~2 GPU-часа",
        "Открытые модели: XLM-R, NLLB, Whisper, ISSAI KazLLM",
        "Код и корпус — открытые"]],
      [fa.FaServer, "Пилот (оценка)", [
        "1 GPU-сервер (T4/L4) для ASR и модели — порядка $400–600 в месяц в облаке",
        "Вычитка казахского фразобанка носителем — ~40 часов",
        "Интеграция с антифродом банка — силами банка"]],
      [fa.FaUserShield, "Команда и запрос", [
        "[ФИО] — ML, данные, разработка (SDU University)",
        "Нужны: банк-партнёр с обезличенными звонками, эксперт антифрода, наставник",
        "Партнёр конкурса ISSAI — казахский ASR и LLM"]],
    ];
    for (let i = 0; i < cols.length; i++) {
      const [ic, h, items] = cols[i];
      const x = 0.6 + i * 4.1;
      s.addShape("roundRect", { x, y: 1.45, w: 3.85, h: 4.55, fill: { color: C.navy2 }, line: { color: C.navy2 }, rectRadius: 0.08 });
      await iconCircle(s, ic, x + 0.25, 1.65, 0.75, C.gold, C.navy);
      T(s, h, { x: x + 1.15, y: 1.8, w: 2.6, h: 0.5, fontSize: 20, bold: true, color: C.white });
      T(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
        { x: x + 0.25, y: 2.6, w: 3.4, h: 3.3, fontSize: 14, color: "DCE6F5", paraSpaceAfter: 8 });
    }
    T(s, "Қалқан — предупреждение, пока деньги ещё у клиента.", { x: 0.6, y: 6.25, w: 12.1, h: 0.6, fontSize: 24,
      bold: true, color: C.gold });
    s.addNotes("Прототип сделан без затрат на вычисления — бесплатные GPU Kaggle и открытые модели. Для пилота нужен " +
      "один GPU-сервер, вычитка казахского носителем и, главное, банк-партнёр с обезличенными звонками. " +
      "Қалқан — это предупреждение, пока деньги ещё у клиента. Спасибо, готов ответить на вопросы.");
  }

  await pres.writeFile({ fileName: OUT });
  console.log("written", OUT);
})();
