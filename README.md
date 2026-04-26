# Analytical AI Assistantr

A domain-specific analytics system designed to replace manual spreadsheet 
analysis with a structured dashboard, a conversational AI assistant, and an automated 
data pipeline - built for small and medium-sized businesses.

The system pulls operational data from Google Sheets, validates and stores it in a 
PostgreSQL database, computes monthly financial and operational metrics, and exposes everything 
through a Streamlit web app where users can explore dashboards, generate monthly 
reports, and ask questions in natural language.

---

## What it does

**Data pipeline** - reads daily revenue, expenses, staff hours, and service counts 
from Google Sheets, validates and normalizes the data, anonymizes personal names 
via opaque identifiers, writes to PostgreSQL, recomputes monthly KPIs.

**Analytics engine** - calculates P&L, Cash Flow, and 
operational metrics.

**Dashboard** - static KPI cards and interactive Plotly charts for any selected 
month. Displays financial and operational data without interpretation - analytical 
conclusions are delegated to the report and chat modules.

**Monthly report generation** - the system collects the full financial context for 
a selected month, builds a structured prompt, and generates a narrative analytical 
report via Claude Sonnet. The report covers P&L, cash flow, staff efficiency, 
service distribution, and expense structure. Exportable as HTML. Average generation 
time ~40 seconds, average cost ~$0.046 per report.

**AI assistant (chat)** — a ReAct agent built with LangGraph, backed by Claude 
Sonnet 4.6, that answers financial and operational questions in natural language. 
The agent has access to 18 specialized tools covering P&L, cash flow, staff metrics, 
trend analysis, anomaly detection, two-month comparisons, and more. Read-only by 
design — the agent never modifies data. Average cost per interaction cycle ~$0.12.
---

## Evaluation summary

The system was evaluated across all three modules:

- **Data pipeline** - 7/7 validation checks passed (100% accuracy)
- **Report generation** - tested on 7 months, 100% template compliance, average LLM-as-a-judge score 4.14/5
- **AI assistant** - 56 test cases across 7 categories and 3 difficulty levels,
  overall correctness 0.879, faithfulness 0.94, tool success rate 1.0

---

## Tech stack

| Layer | What's used |
|---|---|
| Frontend | Streamlit, Plotly |
| Backend | Python 3.11+, pandas, numpy |
| Database | PostgreSQL via Supabase |
| AI / Agent | Claude (Anthropic API) or Ollama · LangChain · LangGraph |
| Data ingestion | Google Sheets API via gspread |
| Auth | bcrypt |
| Export | HTML |

---

## Project structure

```
├── agent/
│   ├── app.py            # Streamlit app (dashboard, chat, user management)
│   ├── graph.py          # LangGraph ReAct agent definition
│   ├── tools.py          # agent tools
│   ├── llm.py            # LLM configuration (Claude / Ollama)
│   ├── system_prompt.py  # Agent instructions and safety rules
│   └── prompt.py         # Report generation template
├── analytics/
│   ├── compute.py        # Monthly metrics orchestration
│   ├── pnl.py            # P&L computation
│   ├── cashflow.py       # Cash flow computation
│   ├── operational.py    # Staff and service metrics
│   ├── queries.py        # Database reads
│   └── metrics_writer.py # Upserts results to monthly_metrics
├── config/
│   ├── settings.py       # Date format constants
│   ├── constants.py      # Metric names and event registry
│   └── supabase.py       # Supabase client setup
├── db/
│   ├── schema.sql        # PostgreSQL schema
│   └── identity.py       # Name anonymization helpers
├── ingestion/            # Google Sheets loaders
├── processing/           # Data cleaning, normalization, validation
├── storage/              # Database writers
├── tests/                # Test suites for pipeline, metrics, chat, reports
├── main.py               # Pipeline entry point
└── run_report_tests.py   # Report generation test harness
```

---

## Database schema

Ten tables:

- `users` - authentication
- `persons` - maps real names to anonymous IDs
- `daily_revenue` - cash and card revenue by day
- `expenses` - operational expense categories
- `amortization` - asset depreciation
- `specialist_capacity` - available hours per specialist per month
- `specialist_activity` - service units by type (massage / physical therapy / physio)
- `specialist_payouts` - salary payouts with generated revenue
- `monthly_metrics` - pre-computed KPI store
- `conversations` - chat session history

---

## Setup

### 1. Clone and install dependencies

```bash
git clone 
cd Analytical-AI-Assistant
pip install -r requirements.txt
```

### 2. Create `.env`

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

ANTHROPIC_API_KEY=sk-ant-...

GOOGLE_CREDENTIALS_PATH=credentials.json
```

### 3. Set up the database

Run the schema against your Supabase project:

```bash
psql your-connection-string -f db/schema.sql
```

### 4. Run the data pipeline

Pulls from Google Sheets, cleans data, writes to the database, computes metrics:

```bash
python main.py
```

### 5. Start the web app

```bash
streamlit run agent/app.py
```
---
