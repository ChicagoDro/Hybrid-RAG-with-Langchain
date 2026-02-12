"""
Graph Retrieval Utilities for Chitown Custom Choppers (GraphRAG Upgrade)

This module loads the canonical knowledge graph from SQLite (or fallback JSON)
and exposes helper functions for querying the graph and transaction-level sales.
Graph: departments, employees, sales_metrics (NetworkX). Transaction data:
sales_transactions, products, marketing_campaigns (SQLite).

Typical usage:

    from RAG_chatbot.graph_retrieval import load_graph, get_direct_reports

    G = load_graph()
    reports = get_direct_reports(G, "person_rosa_martinez")
    tx = get_sales_transactions_for_person("person_derek_vaughn")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import networkx as nx
from langsmith import traceable
import sqlite3

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
# LOADING THE GRAPH
# --------------------------------------------------------------------------------------


def load_graph() -> nx.DiGraph:
    """
    Load the knowledge graph from graph_output.json into a NetworkX DiGraph.

    Each node in the graph has:
        - node_id (string)
        - attributes from the original JSON node dict (type, name, etc.)

    Each edge in the graph has:
        - source
        - target
        - 'relation' attribute
        - optional 'metadata' attribute

    Returns:
        nx.DiGraph
    """
    # Prefer SQLite as the canonical store; fall back to JSON if needed
    if SQLITE_DB_PATH.exists():
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()

        G = nx.DiGraph()

        # Departments -> Department nodes
        for dept_id, name in cur.execute(
            "SELECT id, name FROM departments"
        ):
            G.add_node(
                dept_id,
                type="Department",
                name=name,
            )

        # Employees -> Person nodes plus REPORTS_TO and BELONGS_TO_DEPARTMENT edges
        for emp_id, name, role, dept_id, manager_id in cur.execute(
            "SELECT id, name, role, department_id, manager_id FROM employees"
        ):
            G.add_node(
                emp_id,
                type="Person",
                name=name,
                role=role,
                department_id=dept_id,
            )

            # REPORTS_TO edge
            if manager_id:
                G.add_edge(
                    emp_id,
                    manager_id,
                    relation="REPORTS_TO",
                    metadata={},
                )

            # BELONGS_TO_DEPARTMENT edge
            if dept_id:
                G.add_edge(
                    emp_id,
                    dept_id,
                    relation="BELONGS_TO_DEPARTMENT",
                    metadata={},
                )

        # Sales metrics -> SalesMetric nodes + HAS_SALES_METRIC edges
        for metric_id, person_id, year, month, amount_usd in cur.execute(
            "SELECT id, person_id, year, month, amount_usd FROM sales_metrics"
        ):
            G.add_node(
                metric_id,
                type="SalesMetric",
                person_id=person_id,
                year=year,
                month=month,
                amount_usd=amount_usd,
            )
            G.add_edge(
                person_id,
                metric_id,
                relation="HAS_SALES_METRIC",
                metadata={"period": f"{year}-{month}"},
            )

        conn.close()
        return G

    # Legacy / fallback: load from JSON graph_output.json if SQLite is missing
    if not GRAPH_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Graph store not found. Expected either SQLite at {SQLITE_DB_PATH} "
            f"or JSON at {GRAPH_OUTPUT_PATH}. Please run graph_kg_builder.py first."
        )

    with GRAPH_OUTPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_data: List[Dict[str, Any]] = data.get("nodes", [])
    edges_data: List[Dict[str, Any]] = data.get("edges", [])

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes_data:
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **attrs)

    # Add edges with relation + optional metadata
    for edge in edges_data:
        src = edge["source"]
        tgt = edge["target"]
        rel = edge.get("relation", "RELATED_TO")
        metadata = edge.get("metadata", {})
        G.add_edge(src, tgt, relation=rel, metadata=metadata)

    return G


# --------------------------------------------------------------------------------------
# BASIC LOOKUPS
# --------------------------------------------------------------------------------------


def get_node(G: nx.DiGraph, node_id: str) -> Optional[Dict[str, Any]]:
    """Return node attributes for a given node_id, or None if missing."""
    if node_id not in G.nodes:
        return None
    attrs = G.nodes[node_id].copy()
    attrs["id"] = node_id
    return attrs


def find_person_by_name(G: nx.DiGraph, name: str) -> Optional[str]:
    """
    Find a person node by exact name (case-insensitive).
    Returns the node_id or None if not found.
    """
    name_lower = name.lower()
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Person":
            if str(attrs.get("name", "")).lower() == name_lower:
                return node_id
    return None


def list_people(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return a list of all Person nodes (with id and attributes)."""
    result = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Person":
            item = attrs.copy()
            item["id"] = node_id
            result.append(item)
    return result


def list_departments(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return a list of all Department nodes."""
    result = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Department":
            item = attrs.copy()
            item["id"] = node_id
            result.append(item)
    return result


# --------------------------------------------------------------------------------------
# ORG STRUCTURE QUERIES
# --------------------------------------------------------------------------------------


def get_direct_reports(G: nx.DiGraph, manager_id: str) -> List[Dict[str, Any]]:
    """
    Return all Person nodes that have a REPORTS_TO edge pointing to the manager.

    In our graph:
        (employee) -[REPORTS_TO]-> (manager)
    """
    reports: List[Dict[str, Any]] = []
    for src, tgt, attrs in G.in_edges(manager_id, data=True):
        if attrs.get("relation") == "REPORTS_TO":
            node_attrs = G.nodes[src].copy()
            node_attrs["id"] = src
            reports.append(node_attrs)
    return reports


def get_manager(G: nx.DiGraph, person_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the manager (Person node) for the given person_id, if any.

    In our graph:
        (employee) -[REPORTS_TO]-> (manager)
    """
    for src, tgt, attrs in G.out_edges(person_id, data=True):
        if attrs.get("relation") == "REPORTS_TO":
            manager_id = tgt
            result = G.nodes[manager_id].copy()
            result["id"] = manager_id
            return result
    return None


def get_department_for_person(G: nx.DiGraph, person_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the Department node a person belongs to (via BELONGS_TO_DEPARTMENT edge).
    """
    for src, tgt, attrs in G.out_edges(person_id, data=True):
        if attrs.get("relation") == "BELONGS_TO_DEPARTMENT":
            dept_id = tgt
            result = G.nodes[dept_id].copy()
            result["id"] = dept_id
            return result
    return None


def get_department_team(G: nx.DiGraph, department_id: str) -> List[Dict[str, Any]]:
    """
    Return all Person nodes that belong to the given department.
    """
    team: List[Dict[str, Any]] = []
    for src, tgt, attrs in G.in_edges(department_id, data=True):
        if attrs.get("relation") == "BELONGS_TO_DEPARTMENT":
            node_attrs = G.nodes[src].copy()
            node_attrs["id"] = src
            team.append(node_attrs)
    return team


# --------------------------------------------------------------------------------------
# SALES QUERIES
# --------------------------------------------------------------------------------------


def get_sales_metrics_for_person(
    G: nx.DiGraph, person_id: str
) -> List[Dict[str, Any]]:
    """
    Return all SalesMetric nodes for a given person, sorted by (year, month).
    """
    metrics: List[Dict[str, Any]] = []
    for src, tgt, attrs in G.out_edges(person_id, data=True):
        if attrs.get("relation") == "HAS_SALES_METRIC":
            metric_attrs = G.nodes[tgt].copy()
            metric_attrs["id"] = tgt
            metrics.append(metric_attrs)

    # Sort by (year, month)
    def sort_key(m: Dict[str, Any]) -> Tuple[int, int]:
        year = int(m.get("year", 0))
        month = int(m.get("month", 0))
        return (year, month)

    metrics.sort(key=sort_key)
    return metrics


def get_total_sales_for_person(
    G: nx.DiGraph, person_id: str
) -> float:
    """
    Sum all sales metrics for the given person.
    """
    metrics = get_sales_metrics_for_person(G, person_id)
    return float(sum(m.get("amount_usd", 0.0) for m in metrics))


def get_total_sales_for_department(
    G: nx.DiGraph, department_id: str
) -> float:
    """
    Sum all sales metrics for all people belonging to a department.
    """
    team = get_department_team(G, department_id)
    total = 0.0
    for person in team:
        total += get_total_sales_for_person(G, person["id"])
    return float(total)


@traceable(name="get_q3_2024_total_sales")
def get_q3_2024_total_sales(G: nx.DiGraph) -> float:
    """
    Compute the total Q3 2024 sales across all SalesMetric nodes in the graph.

    This is slightly more generic than summing per person and matches how
    you'd aggregate in a graph database.
    """
    total = 0.0
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "SalesMetric":
            year = attrs.get("year")
            if year == 2024:
                # Our data only covers Q3, but this filter is easy to extend.
                total += float(attrs.get("amount_usd", 0.0))
    return float(total)


# --------------------------------------------------------------------------------------
# SALES TRANSACTIONS (SQLite: products, marketing_campaigns, sales_transactions)
# --------------------------------------------------------------------------------------


def _sales_tables_exist(cur: sqlite3.Cursor) -> bool:
    """Return True if sales_transactions and products exist."""
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sales_transactions','products')"
    )
    return len(cur.fetchall()) >= 2


def get_sales_transactions_for_person(
    person_id: str,
    year: Optional[int] = None,
    month: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Return transaction-level sales for a person from SQLite, with product and
    campaign names. Optionally filter by year and/or month.
    """
    path = db_path or SQLITE_DB_PATH
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if not _sales_tables_exist(cur):
        conn.close()
        return []

    sql = """
        SELECT t.id, t.transaction_date, t.person_id, t.product_id, t.amount_usd,
               t.marketing_campaign_id, t.year, t.month,
               p.name AS product_name, p.category AS product_category,
               c.name AS campaign_name
        FROM sales_transactions t
        JOIN products p ON p.id = t.product_id
        LEFT JOIN marketing_campaigns c ON c.id = t.marketing_campaign_id
        WHERE t.person_id = ?
    """
    params: List[Any] = [person_id]
    if year is not None:
        sql += " AND t.year = ?"
        params.append(year)
    if month is not None:
        sql += " AND t.month = ?"
        params.append(month)
    sql += " ORDER BY t.transaction_date, t.id"

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_sales_by_product_for_person(
    person_id: str,
    year: Optional[int] = None,
    month: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Return sales breakdown by product for a person (total_usd and transaction_count).
    """
    path = db_path or SQLITE_DB_PATH
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    if not _sales_tables_exist(cur):
        conn.close()
        return []

    sql = """
        SELECT p.id AS product_id, p.name AS product_name, p.category AS product_category,
               SUM(t.amount_usd) AS total_usd, COUNT(*) AS transaction_count
        FROM sales_transactions t
        JOIN products p ON p.id = t.product_id
        WHERE t.person_id = ?
    """
    params: List[Any] = [person_id]
    if year is not None:
        sql += " AND t.year = ?"
        params.append(year)
    if month is not None:
        sql += " AND t.month = ?"
        params.append(month)
    sql += " GROUP BY p.id, p.name, p.category ORDER BY total_usd DESC"

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "product_id": r[0],
            "product_name": r[1],
            "product_category": r[2],
            "total_usd": float(r[3]),
            "transaction_count": r[4],
        }
        for r in rows
    ]


def get_sales_by_campaign_for_person(
    person_id: str,
    year: Optional[int] = None,
    month: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Return sales attribution by marketing campaign for a person (total_usd and
    transaction_count). Only transactions with a non-null campaign are included.
    """
    path = db_path or SQLITE_DB_PATH
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    if not _sales_tables_exist(cur):
        conn.close()
        return []

    sql = """
        SELECT c.id AS campaign_id, c.name AS campaign_name,
               SUM(t.amount_usd) AS total_usd, COUNT(*) AS transaction_count
        FROM sales_transactions t
        JOIN marketing_campaigns c ON c.id = t.marketing_campaign_id
        WHERE t.person_id = ? AND t.marketing_campaign_id IS NOT NULL
    """
    params: List[Any] = [person_id]
    if year is not None:
        sql += " AND t.year = ?"
        params.append(year)
    if month is not None:
        sql += " AND t.month = ?"
        params.append(month)
    sql += " GROUP BY c.id, c.name ORDER BY total_usd DESC"

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "campaign_id": r[0],
            "campaign_name": r[1],
            "total_usd": float(r[2]),
            "transaction_count": r[3],
        }
        for r in rows
    ]


def list_products(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all products from SQLite (id, name, category)."""
    path = db_path or SQLITE_DB_PATH
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    if not _sales_tables_exist(cur):
        conn.close()
        return []

    cur.execute("SELECT id, name, category FROM products ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "category": r[2]} for r in rows]


def list_marketing_campaigns(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all marketing campaigns from SQLite (id, name, description)."""
    path = db_path or SQLITE_DB_PATH
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='marketing_campaigns'"
    )
    if not cur.fetchone():
        conn.close()
        return []

    cur.execute("SELECT id, name, description FROM marketing_campaigns ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "description": r[2] or ""} for r in rows]


# --------------------------------------------------------------------------------------
# HIGH-LEVEL SUMMARIES (USEFUL FOR LLM PROMPTS)
# --------------------------------------------------------------------------------------


@traceable(name="format_org_summary")
def format_org_summary(G: nx.DiGraph) -> str:
    """
    Produce a human-readable summary of the org structure suitable for
    inclusion in an LLM context window.
    """
    lines: List[str] = []

    # CEO
    ceos = [
        (nid, attrs)
        for nid, attrs in G.nodes(data=True)
        if attrs.get("type") == "Person" and attrs.get("role", "").upper() == "CEO"
    ]
    for nid, attrs in ceos:
        lines.append(f"CEO: {attrs.get('name')} (id={nid})")

        # Direct reports to CEO (directors)
        reports = get_direct_reports(G, nid)
        for r in reports:
            lines.append(f"  Director: {r['name']} — {r.get('role')} (id={r['id']})")

    # Departments and their members
    lines.append("\nDepartments:")
    for dept in list_departments(G):
        lines.append(f"- {dept['name']} (id={dept['id']})")
        team = get_department_team(G, dept["id"])
        for member in team:
            lines.append(f"    * {member['name']} — {member.get('role')} (id={member['id']})")

    return "\n".join(lines)


@traceable(name="format_sales_overview")
def format_sales_overview(
    G: nx.DiGraph,
    include_transaction_summary: bool = True,
    db_path: Optional[Path] = None,
) -> str:
    """
    Produce a company-wide sales overview for Q3 2024: total sales, per-person
    totals, and optionally by department and transaction-level summary.
    Used when the user asks about sales figures, revenue, or performance in general.
    """
    total_q3 = get_q3_2024_total_sales(G)
    lines = [
        f"Q3 2024 total sales (all employees): ${total_q3:,.2f}",
        "",
        "Sales by person (Q3 2024):",
    ]
    people = list_people(G)
    for person in people:
        pid = person["id"]
        person_total = get_total_sales_for_person(G, pid)
        if person_total > 0:
            lines.append(f"  - {person['name']} ({person.get('role', '')}): ${person_total:,.2f}")
    lines.append("")
    lines.append("Sales by department (Q3 2024):")
    for dept in list_departments(G):
        dept_total = get_total_sales_for_department(G, dept["id"])
        if dept_total > 0:
            lines.append(f"  - {dept['name']}: ${dept_total:,.2f}")

    if include_transaction_summary:
        path = db_path or SQLITE_DB_PATH
        if path.exists():
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            if _sales_tables_exist(cur):
                cur.execute(
                    """
                    SELECT p.name, SUM(t.amount_usd) AS total
                    FROM sales_transactions t
                    JOIN products p ON p.id = t.product_id
                    GROUP BY p.id ORDER BY total DESC LIMIT 5
                    """
                )
                product_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT c.name, SUM(t.amount_usd) AS total
                    FROM sales_transactions t
                    JOIN marketing_campaigns c ON c.id = t.marketing_campaign_id
                    WHERE t.marketing_campaign_id IS NOT NULL
                    GROUP BY c.id ORDER BY total DESC LIMIT 5
                    """
                )
                campaign_rows = cur.fetchall()
                conn.close()
                if product_rows:
                    lines.append("")
                    lines.append("Top product categories (Q3 2024):")
                    for name, amt in product_rows:
                        lines.append(f"  - {name}: ${float(amt):,.2f}")
                if campaign_rows:
                    lines.append("")
                    lines.append("Sales attributed to marketing campaigns (Q3 2024):")
                    for name, amt in campaign_rows:
                        lines.append(f"  - {name}: ${float(amt):,.2f}")

    return "\n".join(lines)


@traceable(name="format_person_sales_summary")
def format_person_sales_summary(
    G: nx.DiGraph,
    person_id: str,
    include_transaction_breakdown: bool = True,
    db_path: Optional[Path] = None,
) -> str:
    """
    Produce a text summary of an employee's Q3 2024 sales performance.
    When transaction data exists in SQLite, includes breakdown by product
    and attribution to marketing campaigns.
    """
    person = get_node(G, person_id)
    if not person:
        return f"No data found for person_id={person_id}"

    metrics = get_sales_metrics_for_person(G, person_id)
    if not metrics:
        return f"No sales metrics found for {person['name']}."

    lines = [f"Sales summary for {person['name']} ({person_id}):"]
    total = 0.0
    for m in metrics:
        year = m.get("year")
        month = m.get("month")
        amt = float(m.get("amount_usd", 0.0))
        total += amt
        lines.append(f"  - {year}-{month}: ${amt:,.2f}")
    lines.append(f"Total Q3 2024 sales: ${total:,.2f}")

    if include_transaction_breakdown:
        by_product = get_sales_by_product_for_person(person_id, db_path=db_path)
        if by_product:
            lines.append("\nBreakdown by product:")
            for row in by_product:
                lines.append(
                    f"  - {row['product_name']} ({row['product_category']}): "
                    f"${row['total_usd']:,.2f} ({row['transaction_count']} tx)"
                )
        by_campaign = get_sales_by_campaign_for_person(person_id, db_path=db_path)
        if by_campaign:
            lines.append("\nAttribution to marketing campaigns:")
            for row in by_campaign:
                lines.append(
                    f"  - {row['campaign_name']}: "
                    f"${row['total_usd']:,.2f} ({row['transaction_count']} tx)"
                )

    return "\n".join(lines)
