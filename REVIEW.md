# Data Warehouse Project Review

## 1. Executive Summary

This project implements a Data Warehouse solution designed to consolidate data from multiple organizational departments (Business, Customer Management, Enterprise, Marketing) into a centralized analytics repository.

The system utilizes a standard **ELT (Extract, Load, Transform)** architecture:
1.  **Extract & Load**: Raw data files (CSV, Parquet, JSON, Excel, etc.) are ingested into a PostgreSQL database.
2.  **Transform**: SQL scripts process this raw data through a "Silver" layer (validation & cleaning) and finally into a "Gold" layer (Star Schema for reporting).

**Current Status:**
The core logic for ingestion and transformation is present and functional. However, the project lacks a unified orchestration layer, meaning the end-to-end process is manual and error-prone. There is no automated workflow to ensure data is loaded and transformed in the correct order.

**Recommendation:**
The project is a strong candidate for migration to **Kestra**, an orchestration platform. Kestra will automate the manual steps, handle dependencies, and provide visibility into pipeline failures.

---

## 2. Technical Overview

### Architecture

*   **Database**: PostgreSQL 15 (Containerized)
*   **Language**: Python 3.11 (for Ingestion), SQL (for Transformation)
*   **Infrastructure**: Docker & Docker Compose
*   **Data Flow**: `Raw Files` -> `Public Schema` (Raw Tables) -> `Silver Schema` (Validated/Cleaned) -> `Gold Schema` (Dimensional Model)

### Key Components

#### 1. Ingestion (`scripts/ingest/ingest_data.py`)
A flexible Python script that reads various file formats (CSV, Parquet, JSON, Excel, HTML) and loads them into Postgres.
*   **Features**: Chunked loading for large files, Postgres `COPY` command support for speed, and basic type mapping.

#### 2. Validation (`scripts/ingest/transform/silver_layer/`)
A series of SQL scripts (e.g., `01_campaign_validation.sql`) that:
*   Read raw data from the `public` schema.
*   Apply data quality rules (null checks, format checks).
*   Split data into `valid` and `invalid` tables in the `silver` schema.
*   Log quality metrics to `silver.dq_summary`.

#### 3. Silver Layer (`scripts/ingest/transform/silver_layer/silver_build.sql`)
Consolidates the `valid` data into clean, normalized tables. This layer acts as the "Single Source of Truth" before modeling.

#### 4. Gold Layer (`scripts/ingest/transform/gold_layer.sql`)
Transforms the Silver data into a Star Schema optimized for analytics:
*   **Dimension Tables**: `dim_date`, `dim_merchant`, `dim_product`, `dim_user`, etc.
*   **Fact Tables**: `fact_order`, `fact_line_item`.

---

## 3. Issues & Technical Debt

### Critical Issues
1.  **No Orchestration**: There is no script or tool to run the pipeline end-to-end. A user must manually run `ingest_data.py` for every single file (dozens of times) and then run the SQL scripts in a specific order.
2.  **Implicit Dependencies**: The SQL scripts assume specific table names exist in the `public` schema. If the ingestion script is run with a typo in the table name, the validation scripts will fail silently or throw SQL errors.
3.  **Error Handling**: If a transformation step fails, there is no mechanism to stop the subsequent steps or alert the user.

### Technical Debt
1.  **Hardcoded SQL**: Transformation logic is buried in large SQL files. While common for ELT, it makes unit testing difficult.
2.  **Lack of Documentation**: The `README.md` is empty. A new developer would not know how to bring up the environment or run the ETL.
3.  **Data Quality Visibility**: While DQ metrics are captured in `silver.dq_summary`, there is no dashboard or alert system to surface these issues to stakeholders.

---

## 4. Migration to Kestra

The migration to Kestra is straightforward and will provide immediate value.

### Proposed Kestra Workflow

We can structure the Kestra flow into three main phases, mirroring the current architecture:

#### Phase 1: Parallel Ingestion
Use Kestra's **Parallel** task to ingest files concurrently.
*   **Task**: `io.kestra.plugin.scripts.python.Script`
*   **Action**: Execute `ingest_data.py` for each file in `data/raw`.
*   **Inputs**: File path, target table name (e.g., `public.campaign_data_raw`).

#### Phase 2: Sequential Validation
Once ingestion is complete, run the validation SQL scripts.
*   **Task**: `io.kestra.plugin.jdbc.postgresql.Query`
*   **Action**: Execute files like `01_campaign_validation.sql`.
*   **Benefit**: If validation fails (e.g., too many invalid rows), Kestra can halt the pipeline.

#### Phase 3: Layer Building
Run the Silver and Gold build scripts.
*   **Task**: `io.kestra.plugin.jdbc.postgresql.Query`
*   **Action**: Execute `silver_build.sql` then `gold_layer.sql`.

### Automation & Triggers
*   **Schedule**: Run the pipeline daily/hourly.
*   **File Watcher**: Trigger the pipeline automatically when new files land in the `data/raw` directory.

---

## 5. How to Run (Current State)

Since there is no orchestration, you must currently run the project manually:

1.  **Start Infrastructure**:
    ```bash
    docker-compose up -d
    ```

2.  **Ingest Data** (Example for one file):
    ```bash
    docker-compose run --rm etl \
      --path /data/raw/Marketing\ Department/campaign_data.csv \
      --table_name campaign_data_raw \
      --user postgres --password postgres --host db --port 5432 --db bronze
    ```
    *Repeat this for every file in `data/raw`.*

3.  **Run Transformations**:
    You need to execute the SQL scripts against the database.
    ```bash
    # Run Validation
    cat scripts/ingest/transform/silver_layer/01_campaign_validation.sql | docker exec -i dmaw_db psql -U postgres -d bronze

    # ... Run all other validations ...

    # Run Silver Build
    cat scripts/ingest/transform/silver_layer/silver_build.sql | docker exec -i dmaw_db psql -U postgres -d bronze

    # Run Gold Build
    cat scripts/ingest/transform/gold_layer.sql | docker exec -i dmaw_db psql -U postgres -d bronze
    ```
