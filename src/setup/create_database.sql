PRAGMA foreign_keys = ON;

-- Drop existing tables if they exist (for easy rebuilds in dev)
DROP TABLE IF EXISTS sales_transactions;
DROP TABLE IF EXISTS sales_metrics;
DROP TABLE IF EXISTS marketing_campaigns;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;

-- Departments (e.g., Custom Builds, Repairs, Retail & Merch)
CREATE TABLE departments (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

-- People / employees (including CEO and staff)
CREATE TABLE employees (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL,
    department_id TEXT REFERENCES departments(id),
    manager_id    TEXT REFERENCES employees(id)
);

-- Monthly sales metrics (Q3 2024 in the current dataset)
CREATE TABLE sales_metrics (
    id         TEXT PRIMARY KEY,
    person_id  TEXT NOT NULL REFERENCES employees(id),
    year       INTEGER NOT NULL,
    month      TEXT NOT NULL, -- zero-padded string (e.g., '07', '08')
    amount_usd REAL NOT NULL
);

-- Products (aligned to Product_* and repair/custom build docs)
CREATE TABLE products (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL  -- e.g. custom_build, parts, labor, accessories
);

-- Marketing campaigns (aligned to Marketing_* PDFs)
CREATE TABLE marketing_campaigns (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT
);

-- Transaction-level sales; sums per (person_id, year, month) match sales_metrics.amount_usd
CREATE TABLE sales_transactions (
    id                   TEXT PRIMARY KEY,
    transaction_date     TEXT NOT NULL,  -- ISO date (YYYY-MM-DD)
    person_id            TEXT NOT NULL REFERENCES employees(id),
    product_id           TEXT NOT NULL REFERENCES products(id),
    amount_usd           REAL NOT NULL,
    marketing_campaign_id TEXT REFERENCES marketing_campaigns(id),  -- lead driver, nullable
    year                 INTEGER NOT NULL,
    month                TEXT NOT NULL
);

