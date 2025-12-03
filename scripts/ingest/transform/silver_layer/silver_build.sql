CREATE SCHEMA IF NOT EXISTS silver;

-------------------------------------------------------------
-- 1. CAMPAIGN DATA 
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.campaign_data (
    campaign_id          text,
    campaign_name        text,
    campaign_description text,
    discount             text
);

TRUNCATE TABLE silver.campaign_data;

WITH cleaned AS (
    SELECT
        TRIM(campaign_id)          AS campaign_id,
        TRIM(campaign_name)        AS campaign_name,
        TRIM(campaign_description) AS campaign_description,
        TRIM(discount)             AS discount_raw
    FROM silver.valid_campaign_data_raw
)
INSERT INTO silver.campaign_data (
    campaign_id,
    campaign_name,
    campaign_description,
    discount
)
SELECT
    campaign_id,
    campaign_name,
    campaign_description,
    REGEXP_REPLACE(discount_raw, '[^0-9]', '', 'g') AS discount
FROM cleaned;

-------------------------------------------------------------
-- 2. MERCHANT DATA
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.merchant_data (
    merchant_id    text,
    creation_date  text,
    merchant_name  text,
    street         text,
    state          text,
    city           text,
    country        text,
    contact_number text
);

TRUNCATE TABLE silver.merchant_data;

INSERT INTO silver.merchant_data (
    merchant_id,
    creation_date,
    merchant_name,
    street,
    state,
    city,
    country,
    contact_number
)
SELECT
    TRIM(merchant_id)    AS merchant_id,
    TRIM(creation_date)  AS creation_date,
    TRIM(name)           AS merchant_name,
    TRIM(street)         AS street,
    TRIM(state)          AS state,
    TRIM(city)           AS city,
    TRIM(country)        AS country,
    TRIM(contact_number) AS contact_number
FROM silver.valid_merchant_data_raw;

-------------------------------------------------------------
-- 3. STAFF DATA
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.staff_data (
    staff_id       text,
    staff_name     text,
    job_level      text,
    street         text,
    state          text,
    city           text,
    country        text,
    contact_number text,
    creation_date  text
);

TRUNCATE TABLE silver.staff_data;

INSERT INTO silver.staff_data (
    staff_id,
    staff_name,
    job_level,
    street,
    state,
    city,
    country,
    contact_number,
    creation_date
)
SELECT
    TRIM(staff_id)       AS staff_id,
    TRIM(name)           AS staff_name,
    TRIM(job_level)      AS job_level,
    TRIM(street)         AS street,
    TRIM(state)          AS state,
    TRIM(city)           AS city,
    TRIM(country)        AS country,
    TRIM(contact_number) AS contact_number,
    TRIM(creation_date)  AS creation_date
FROM silver.valid_staff_data_raw;

-------------------------------------------------------------
-- 4. USER JOB
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.user_job (
    user_id   text,
    name      text,
    job_title text,
    job_level text
);

TRUNCATE TABLE silver.user_job;

INSERT INTO silver.user_job (
    user_id,
    name,
    job_title,
    job_level
)
SELECT
    TRIM(user_id)   AS user_id,
    TRIM(name)      AS name,
    TRIM(job_title) AS job_title,
    TRIM(job_level) AS job_level
FROM silver.valid_user_job_raw;

-------------------------------------------------------------
-- 5. USER DATA
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.user_data (
    user_id        text,
    creation_date  text,
    name           text,
    street         text,
    state          text,
    city           text,
    country        text,
    birthdate      text,
    gender         text,
    device_address text,
    user_type      text
);

TRUNCATE TABLE silver.user_data;

INSERT INTO silver.user_data (
    user_id,
    creation_date,
    name,
    street,
    state,
    city,
    country,
    birthdate,
    gender,
    device_address,
    user_type
)
SELECT
    TRIM(user_id)        AS user_id,
    TRIM(creation_date)  AS creation_date,
    TRIM(name)           AS name,
    TRIM(street)         AS street,
    TRIM(state)          AS state,
    TRIM(city)           AS city,
    TRIM(country)        AS country,
    TRIM(birthdate)      AS birthdate,
    TRIM(gender)         AS gender,
    TRIM(device_address) AS device_address,
    TRIM(user_type)      AS user_type
FROM silver.valid_user_data_raw;

-------------------------------------------------------------
-- 6. USER CREDIT CARD
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.user_credit_card (
    user_id            text,
    name               text,
    issuing_bank       text,
    credit_card_number bigint
);

TRUNCATE TABLE silver.user_credit_card;

INSERT INTO silver.user_credit_card (
    user_id,
    name,
    issuing_bank,
    credit_card_number
)
SELECT
    TRIM(user_id)      AS user_id,
    TRIM(name)         AS name,
    TRIM(issuing_bank) AS issuing_bank,
    credit_card_number
FROM silver.valid_user_credit_card_raw;

-------------------------------------------------------------
-- 7. PRODUCT LIST
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.product_list (
    product_id   text,
    product_name text,
    product_type text,
    price        double precision
);

TRUNCATE TABLE silver.product_list;

INSERT INTO silver.product_list (
    product_id,
    product_name,
    product_type,
    price
)
SELECT
    TRIM(product_id)   AS product_id,
    TRIM(product_name) AS product_name,
    TRIM(product_type) AS product_type,
    price::double precision AS price
FROM silver.valid_product_list_raw;

-------------------------------------------------------------
-- 8. ORDER DELAYS
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.order_delays (
    order_id   text,
    delay_days integer
);

TRUNCATE TABLE silver.order_delays;

INSERT INTO silver.order_delays (
    order_id,
    delay_days
)
SELECT
    TRIM(order_id) AS order_id,
    COALESCE(
        NULLIF(
            REGEXP_REPLACE("delay in days"::text, '[^0-9]', '', 'g'),
            ''
        ),
        '0'
    )::int AS delay_days
FROM silver.valid_order_delays_raw;

-------------------------------------------------------------
-- 9. ORDERS (UNION OF ALL ORDER DATA)
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.orders (
    order_id          text,
    user_id           text,
    estimated_arrival text,
    transaction_date  text
);

TRUNCATE TABLE silver.orders;

INSERT INTO silver.orders (
    order_id,
    user_id,
    estimated_arrival,
    transaction_date
)
SELECT
    order_id,
    user_id,
    "estimated arrival" AS estimated_arrival,
    transaction_date
FROM silver.valid_order_data_20200101_20200701_raw

UNION ALL
SELECT
    order_id,
    user_id,
    "estimated arrival",
    transaction_date
FROM silver.valid_order_data_20200701_20211001_raw

UNION ALL
SELECT
    order_id,
    user_id,
    "estimated arrival",
    transaction_date
FROM silver.valid_order_data_20211001_20220101_raw

UNION ALL
SELECT
    order_id,
    user_id,
    "estimated arrival",
    transaction_date
FROM silver.valid_order_data_20220101_20221201_raw

UNION ALL
SELECT
    order_id,
    user_id,
    "estimated arrival",
    transaction_date
FROM silver.valid_order_data_20221201_20230601_raw

UNION ALL
SELECT
    order_id,
    user_id,
    "estimated arrival",
    transaction_date
FROM silver.valid_order_data_20230601_20240101_raw;

-------------------------------------------------------------
-- 10. ORDER_WITH_MERCHANT
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.order_with_merchant (
    order_id    text,
    merchant_id text,
    staff_id    text
);

TRUNCATE TABLE silver.order_with_merchant;

INSERT INTO silver.order_with_merchant (
    order_id,
    merchant_id,
    staff_id
)
SELECT
    order_id,
    merchant_id,
    staff_id
FROM silver.valid_order_with_merchant_data1_raw

UNION ALL
SELECT
    order_id,
    merchant_id,
    staff_id
FROM silver.valid_order_with_merchant_data2_raw

UNION ALL
SELECT
    order_id,
    merchant_id,
    staff_id
FROM silver.valid_order_with_merchant_data3_raw;

-------------------------------------------------------------
-- 11. LINE ITEM PRICES
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.line_item_prices (
    order_id text,
    price    double precision,
    quantity text
);

TRUNCATE TABLE silver.line_item_prices;

INSERT INTO silver.line_item_prices (
    order_id,
    price,
    quantity
)
SELECT
    TRIM(order_id) AS order_id,
    price::double precision AS price,
    TRIM(quantity) AS quantity
FROM silver.valid_line_item_prices1_raw

UNION ALL
SELECT
    TRIM(order_id),
    price::double precision,
    TRIM(quantity)
FROM silver.valid_line_item_prices2_raw

UNION ALL
SELECT
    TRIM(order_id),
    price::double precision,
    TRIM(quantity)
FROM silver.valid_line_item_prices3_raw;

-------------------------------------------------------------
-- 12. LINE ITEM PRODUCTS
-------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.line_item_products (
    order_id     text,
    product_id   text,
    product_name text
);

TRUNCATE TABLE silver.line_item_products;

INSERT INTO silver.line_item_products (
    order_id,
    product_id,
    product_name
)
SELECT
    TRIM(order_id)   AS order_id,
    TRIM(product_id) AS product_id,
    TRIM(product_name) AS product_name
FROM silver.valid_line_item_products1_raw

UNION ALL
SELECT
    TRIM(order_id),
    TRIM(product_id),
    TRIM(product_name)
FROM silver.valid_line_item_products2_raw

UNION ALL
SELECT
    TRIM(order_id),
    TRIM(product_id),
    TRIM(product_name)
FROM silver.valid_line_item_products3_raw;
