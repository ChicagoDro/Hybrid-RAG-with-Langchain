PYTHON ?= python
SQLITE3 ?= sqlite3

PROJECT_ROOT := $(shell pwd)
INDICES_DIR := $(PROJECT_ROOT)/indices
DATA_DIR := $(PROJECT_ROOT)/data/Chitown_Custom_Choppers
DB_PATH := $(DATA_DIR)/chitown_graph.db
CREATE_DB_SQL := $(PROJECT_ROOT)/src/setup/create_database.sql
SEED_DATA_SQL := $(PROJECT_ROOT)/src/setup/seed_data.sql
DROP_DB_SQL := $(PROJECT_ROOT)/src/setup/drop_database.sql

.PHONY: all clean index app db graph-db libs

# Run full workflow: rebuild graph DB, index, then start Streamlit app
all: graph-db index app

## clean: remove generated indices and database
clean:
	rm -rf "$(INDICES_DIR)"
	rm -f "$(DB_PATH)"

## libs: install/upgrade Python dependencies from requirements.txt
libs:
	pip install --upgrade -r requirements.txt

## db: create and seed the SQLite graph database
db: $(CREATE_DB_SQL) $(SEED_DATA_SQL)
	@echo "Creating SQLite graph database..."
	$(SQLITE3) "$(DB_PATH)" < "$(CREATE_DB_SQL)"
	$(SQLITE3) "$(DB_PATH)" < "$(SEED_DATA_SQL)"
	@echo "Database created and seeded: $(DB_PATH)"

## graph-db: build graph from SQLite and write graph_output.json (depends on db)
graph-db: db
	$(PYTHON) src/RAG_build/graph_kg_builder.py

## index: build the FAISS vector index for the current LLM_PROVIDER
index:
	$(PYTHON) src/RAG_build/ingest_embed_index.py

## app: launch the Streamlit chatbot UI (builds index if missing)
app: index
	streamlit run src/RAG_chatbot/streamlit_app.py

