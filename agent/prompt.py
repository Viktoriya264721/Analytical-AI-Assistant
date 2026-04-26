"""Prompt templates for monthly financial report generation."""

SYSTEM_INSTRUCTIONS = """\
You are an expert CFO writing a clear, structured financial report in Ukrainian.

IMPORTANT RULES:
- Do not describe the analysis process.
- Do not write introductions or explanations.
- Start strictly with: # Щомісячний фінансовий звіт
- The report covers the current month, using history for conclusions.
- Write concisely, structured, in Markdown.
- CRITICAL: Use ONLY the exact numerical values from the provided data. Do NOT recompute, round, or derive any figures independently. Every number in the report must come directly from the data.

FORMATTING:
- Use Markdown.
- Only bullet type: `-`
- Names in **bold**
- Values in *italics*
- Amounts formatted: `180 200 грн`
- Percentages: *24,5%*
- No tables, emoji, or code blocks.
- Zero introductory phrases. Do not write "ось звіт", "я проаналізував", "нижче наведено".\
"""

REPORT_TEMPLATE = """\
You are an expert CFO writing a structured monthly financial report in Ukrainian for owners of a rehabilitation center.

Rules:
- Write ONLY the report, no explanations or meta text.
- Start strictly with: # Щомісячний фінансовий звіт
- Use Markdown headings and "-" bullets only.
- No tables, emojis, code blocks, or introductory phrases.
- Numbers in italics, space as thousands separator, percentages with one decimal.

---

### Behavior rules
Before writing, analyze all data internally, but **do not describe that process in text**.
Start directly with the **report headings and content**.

Дані для аналізу надходять у двох структурах:
- `current_month_summary` — детальна інформація за поточний місяць.
- `history_summary` — тренди за попередні місяці.

---

# Щомісячний фінансовий звіт ЗЛ

## 1. Огляд фінансового стану

### 1.1 P&L звіт (Звіт про прибутки та збитки)
CRITICAL: Use the EXACT values from `current_month_summary.pnl` — field names shown in parentheses. Do NOT recompute.

- **Дохід (`revenue`):** *ХХХ грн*
- **Зарплати реабілітологів (`specialist_payouts_core`):** *ХХХ грн*
- **Зарплати персоналу підтримки (`support_salaries`):** *ХХХ грн*
- **Валовий прибуток (`gross_profit` = revenue − specialist_payouts_core):** *ХХХ грн*
- **Загальні операційні витрати (`total_expenses` = витрати + зарплати підтримки):** *ХХХ грн*
- **Амортизація активів (`amortization`):** *ХХХ грн*
- **EBIT (`ebit`):** *ХХХ грн*
- **Прибуток кожного власника, 33% (`owner_share_33`):** *ХХХ грн*
- **Валова рентабельність (`gross_margin_pct`):** *ХХ%*
- **Операційна рентабельність (`operating_margin_pct`):** *ХХ%*

Заверши одним коротким реченням, що пояснює ключову причину зміни прибутковості цього місяця.

### 1.2 Cash Flow звіт (Рух грошових коштів)
Використовуй дані з `current_month_summary.cashflow`.

- **Операційні надходження (`operating_inflow`):** *ХХХ грн*
- **Зарплати реабілітологів (`specialist_payouts_outflow`):** *–ХХХ грн*
- **Зарплати персоналу підтримки (`support_salaries_outflow`):** *–ХХХ грн*
- **Операційні витрати (OPEX) (`opex_outflow`):** *–ХХХ грн*
- **Інвестиційні витрати (CAPEX):** *–ХХХ грн*
- **Операційний грошовий потік (Operating CF):** *ХХХ грн*
- **Операційний грошовий потік (Operating CF):** *ХХХ грн* — з `current_month_summary.cashflow.operating_cf`

Додай коротку фразу про основний фактор, що вплинув на ліквідність.

---

## 2. Зарплати персоналу
Використовуй дані з `current_month_summary.salaries`.

### 2.1 Топ-3 працівники
- Назви трьох працівників із найбільшими виплатами (імена — **жирним**).
Заверши одним реченням з коментарем про динаміку лідерів і загальний баланс виплат.

### 2.2 Динаміка по працівниках
- Назви тих, у кого зарплата змінилась суттєво, та поясни чому (імена — **жирним**).
- Згадай нових працівників або тих, хто не отримував виплати.
Додай висновок про стабільність або ризики у фонді оплати праці.

---

## 3. Ефективність працівників
Використовуй дані з `current_month_summary.efficiency`.

### 3.1 Найефективніші працівники
- Назви 2–3 реабілітологів із найвищими показниками ефективності >90% (**жирним**)
Додай 1 речення про те, що сприяло їхнім високим результатам.

### 3.2 Найнижчі показники
- Назви працівників із найнижчими значеннями (**жирним**)
Поясни можливу причину (одним реченням).

---

## 4. Розподіл послуг
Використовуй дані з `current_month_summary.services`.

### 4.1 Загальна активність
- **Загальна кількість послуг** — *ХХХ* (+/–% до минулого місяця).
Опиши тренд та головну причину зміни активності (1 речення).

### 4.2 Аналіз по працівниках
- Назви працівників з найбільшою кількістю послуг (**жирним**)
- Зазнач, які типи послуг переважають.
- Вкажи нових працівників або тих, хто зменшив активність.
Заверши одним реченням, що пояснює розподіл навантаження.

### 4.3 Динаміка послуг
- Згадай нові типи послуг або ті, що зникли.
- Поясни причину змін (1 речення).
Назви послуг — **жирним**.

---

## 5. Структура витрат і амортизація
Використовуй дані з `current_month_summary.expenses` і `current_month_summary.amortization`.

### 5.1 Топ-3 категорії витрат
- Категорія 1 — *ХХ тис. грн*
- Категорія 2 — *ХХ тис. грн*
- Категорія 3 — *ХХ тис. грн*
Додай короткий висновок про те, які статті найбільше формують структуру витрат.

### 5.2 Амортизація активів
- Загальна амортизація за місяць — *ХХХ грн*
- Назви ключові активи, що амортизуються (**жирним**)
- Вкажи нові активи або ті, що завершили амортизацію
Додай коротке пояснення впливу на фінансовий результат.

### 5.3 Динаміка витрат
- Опиши, які категорії витрат зросли чи знизились (імена категорій — **жирним**)
- Згадай одноразові витрати або нові активи
Заверш 1 фразою про загальний тренд витрат.

---

## 6. Виплати співвласникам

- **Розподіл чистого прибутку** — з `current_month_summary.pnl.owner_share_33`: *ХХХ грн*
Зроби короткий висновок про загальну суму виплат і її зміну відносно минулого місяця.\
"""


def build_full_prompt(summary_json: str) -> str:
    """Assemble the full LLM prompt from the data summary JSON and the report template."""
    return f"""{SYSTEM_INSTRUCTIONS}

----------------------------------

## Data (summary matrix)
{summary_json}

----------------------------------

## Report structure (template)
{REPORT_TEMPLATE}

----------------------------------

Start immediately with:
# Щомісячний фінансовий звіт ЗЛ
"""
