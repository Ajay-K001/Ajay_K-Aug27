# Medallion Pipeline

This repository provides a medallion-style data pipeline skeleton for profiling raw CSV inputs, generating STTM rules, creating Bronze/Silver/Gold layer artifacts, and producing reports.

## Structure

- `agents/` contains the profiler, STTM, Bronze, Silver, Gold, Reporter, and Orchestrator agent modules.
- `core/` contains shared configuration, observability, memory, and pipeline state modules.
- `data/landing/` contains sample CSV inputs such as sales, products, and stores.
- `data/profiles/` receives data profiling JSON output.
- `reports/` receives generated reports.

## Run

```bash
python app/main.py --files data/landing/sales_data.csv data/landing/products.csv data/landing/stores.csv
```

Use the configured API keys in `.env` for the selected LLM provider.
