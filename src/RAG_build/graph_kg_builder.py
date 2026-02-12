"""
Graph KG Builder for Chitown Custom Choppers (GraphRAG Upgrade, Data-Driven)

This script builds a knowledge graph from the SQLite database:

    data/Chitown_Custom_Choppers/chitown_graph.db

Tables used: departments, employees, sales_metrics (see create_database.sql).

It produces a canonical graph representation:

    data/Chitown_Custom_Choppers/graph/graph_output.json

Ensure the database exists and is seeded first (e.g. make db or run
create_database.sql and seed_data.sql).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --------------------------------------------------------------------------------------
# PATH CONFIG (from src.config)
# --------------------------------------------------------------------------------------
import sys
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from src import config

DATA_DIR = config.DATA_DIR
GRAPH_DIR = config.GRAPH_DIR
GRAPH_OUTPUT_PATH = config.GRAPH_OUTPUT_PATH
SQLITE_DB_PATH = config.SQLITE_DB_PATH


# --------------------------------------------------------------------------------------
# GRAPH MODEL HELPERS
# --------------------------------------------------------------------------------------


def make_department_node(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "type": "Department",
        "name": record["name"],
    }


def make_person_node(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "type": "Person",
        "name": record["name"],
        "role": record["role"],
        "department_id": record.get("department_id"),
    }


def make_sales_metric_node(metric_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": metric_id,
        "type": "SalesMetric",
        "person_id": record["person_id"],
        "month": record["month"],
        "year": record["year"],
        "amount_usd": float(record["amount_usd"]),
    }


def make_edge(
    source: str,
    target: str,
    relation: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    edge = {
        "source": source,
        "target": target,
        "relation": relation,
    }
    if metadata:
        edge["metadata"] = metadata
    return edge


# --------------------------------------------------------------------------------------
# LOAD DATA FROM SQLITE
# --------------------------------------------------------------------------------------


def load_from_sqlite(
    db_path: Path | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load departments, employees, and sales_metrics from the SQLite database.
    Returns (departments, employees, sales_metrics) as lists of dicts.
    """
    path = db_path or SQLITE_DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Graph database not found at {path}. "
            "Run 'make db' or create_database.sql + seed_data.sql first."
        )

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    departments_raw: List[Dict[str, Any]] = []
    for row in cur.execute("SELECT id, name FROM departments"):
        departments_raw.append({"id": row["id"], "name": row["name"]})

    employees_raw: List[Dict[str, Any]] = []
    for row in cur.execute(
        "SELECT id, name, role, department_id, manager_id FROM employees"
    ):
        employees_raw.append({
            "id": row["id"],
            "name": row["name"],
            "role": row["role"],
            "department_id": row["department_id"],
            "manager_id": row["manager_id"],
        })

    sales_raw: List[Dict[str, Any]] = []
    for row in cur.execute(
        "SELECT id, person_id, year, month, amount_usd FROM sales_metrics"
    ):
        sales_raw.append({
            "id": row["id"],
            "person_id": row["person_id"],
            "year": row["year"],
            "month": row["month"],
            "amount_usd": row["amount_usd"],
        })

    conn.close()
    return departments_raw, employees_raw, sales_raw


# --------------------------------------------------------------------------------------
# BUILD GRAPH FROM SQLITE DATA
# --------------------------------------------------------------------------------------


def build_graph(db_path: Path | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Construct the graph nodes and edges from the SQLite database
    (departments, employees, sales_metrics tables).
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # ---------------- Load from SQLite ----------------
    departments_raw, employees_raw, sales_raw = load_from_sqlite(db_path)

    # ---------------- Departments ----------------
    dept_nodes = [make_department_node(d) for d in departments_raw]
    nodes.extend(dept_nodes)

    dept_ids = {d["id"] for d in departments_raw}

    # ---------------- People ----------------
    person_nodes = [make_person_node(e) for e in employees_raw]
    nodes.extend(person_nodes)

    # Index employees by id for edge construction
    employees_by_id: Dict[str, Dict[str, Any]] = {e["id"]: e for e in employees_raw}

    # ---------------- Reporting Lines & Department Membership ----------------
    for emp in employees_raw:
        emp_id = emp["id"]
        manager_id = emp.get("manager_id")
        dept_id = emp.get("department_id")

        # REPORTS_TO edge (employee -> manager)
        if manager_id:
            if manager_id not in employees_by_id:
                # In a real system you'd log this somewhere central
                print(f"[WARN] manager_id '{manager_id}' not found for employee '{emp_id}'")
            else:
                edges.append(
                    make_edge(
                        source=emp_id,
                        target=manager_id,
                        relation="REPORTS_TO",
                    )
                )

        # BELONGS_TO_DEPARTMENT edge (employee -> department)
        if dept_id:
            if dept_id not in dept_ids:
                print(f"[WARN] department_id '{dept_id}' not found for employee '{emp_id}'")
            else:
                edges.append(
                    make_edge(
                        source=emp_id,
                        target=dept_id,
                        relation="BELONGS_TO_DEPARTMENT",
                    )
                )

    # ---------------- Sales Metrics ----------------
    sales_nodes: List[Dict[str, Any]] = []

    for s in sales_raw:
        metric_id = s["id"]
        person_id = s["person_id"]
        year = s["year"]
        month = str(s["month"])
        amount = float(s["amount_usd"])

        metric_node = make_sales_metric_node(
            metric_id,
            {
                "person_id": person_id,
                "year": year,
                "month": month,
                "amount_usd": amount,
            },
        )
        sales_nodes.append(metric_node)

        edges.append(
            make_edge(
                source=person_id,
                target=metric_id,
                relation="HAS_SALES_METRIC",
                metadata={"period": f"{year}-{month}"},
            )
        )

    nodes.extend(sales_nodes)

    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    graph_data = build_graph()

    # Write graph as JSON for inspection / backward compatibility with graph_retrieval fallback
    with GRAPH_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph built from SQLite: {SQLITE_DB_PATH}")
    print(f"Graph JSON written to: {GRAPH_OUTPUT_PATH}")
    print(f"Nodes: {len(graph_data['nodes'])}, Edges: {len(graph_data['edges'])}")


if __name__ == "__main__":
    main()
