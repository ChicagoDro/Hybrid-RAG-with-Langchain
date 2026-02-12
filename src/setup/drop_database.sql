-- Drop all tables (SQLite doesn't support DROP DATABASE)
-- This script is used before recreating the schema
-- Note: The actual database file deletion is handled by the Makefile

DROP TABLE IF EXISTS sales_transactions;
DROP TABLE IF EXISTS sales_metrics;
DROP TABLE IF EXISTS marketing_campaigns;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;
