CREATE SCHEMA IF NOT EXISTS gold;

-- ============================================================
-- 1. DIM_DATE
-- ============================================================
DROP TABLE IF EXISTS gold.dim_date CASCADE;
CREATE TABLE gold.dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE,
    year INT,
    month INT,
    day INT
);

INSERT INTO gold.dim_date
SELECT 
    TO_CHAR(datum, 'yyyymmdd')::INT AS date_key,
    datum AS full_date,
    EXTRACT(YEAR FROM datum) AS year,
    EXTRACT(MONTH FROM datum) AS month,
    EXTRACT(DAY FROM datum) AS day
FROM (SELECT '1970-01-01'::DATE + SEQUENCE.DAY AS datum
      FROM GENERATE_SERIES(0, 365*100) AS SEQUENCE(DAY)
     ) DQ
ORDER BY 1;


-- ============================================================
-- 2. DIM_MERCHANT
-- ============================================================
DROP TABLE IF EXISTS gold.dim_merchant CASCADE;
CREATE TABLE gold.dim_merchant (
    merchant_key SERIAL PRIMARY KEY,
    merchant_id INT,
    merchant_creation_date_key INT,
    merchant_name VARCHAR(100),
    merchant_street VARCHAR(100),
    merchant_state VARCHAR(50),
    merchant_city VARCHAR(50),
    merchant_country VARCHAR(100),
    merchant_contact_number VARCHAR(20)
);

INSERT INTO gold.dim_merchant (
	merchant_id, 
	merchant_creation_date_key, 
	merchant_name, 
	merchant_street, 
	merchant_state, 
	merchant_city, 
	merchant_country, 
	merchant_contact_number
)
SELECT 
    m.merchant_id::INT,
    d.date_key,
    m.name,
    m.street,
    m.state,
    m.city,
    m.country,
    m.contact_number
FROM silver.merchant_data m
LEFT JOIN gold.dim_date d ON m.creation_date = d.full_date;


-- ============================================================
-- 3. DIM_CAMPAIGN
-- ============================================================
DROP TABLE IF EXISTS gold.dim_campaign CASCADE;
CREATE TABLE gold.dim_campaign (
    campaign_key SERIAL PRIMARY KEY,
    campaign_id INT,
    campaign_name VARCHAR(100),
    campaign_description VARCHAR(500),
    campaign_discount VARCHAR(50)
);

INSERT INTO gold.dim_campaign (
	campaign_id, 
	campaign_name, 
	campaign_description, 
	campaign_discount
)
SELECT 
    campaign_id::INT,
    campaign_name,
    campaign_description,
    discount
FROM silver.campaign_data;


-- ============================================================
-- 4. DIM_PRODUCT
-- ============================================================
DROP TABLE IF EXISTS gold.dim_product CASCADE;
CREATE TABLE gold.dim_product (
    product_key SERIAL PRIMARY KEY,
    product_id INT,
    product_name VARCHAR(100),
    product_type VARCHAR(50),
    product_price DECIMAL(10,2)
);

INSERT INTO gold.dim_product (
	product_id, 
	product_name,
	product_type, 
	product_price
)
SELECT 
    product_id::INT,
    product_name,
    product_type,
    price
FROM silver.product_list;


-- ============================================================
-- 5. DIM_STAFF
-- ============================================================
DROP TABLE IF EXISTS gold.dim_staff CASCADE;
CREATE TABLE gold.dim_staff (
    staff_key SERIAL PRIMARY KEY,
    staff_id INT,
    staff_creation_date_key INT,
    staff_name VARCHAR(100),
    staff_job_level VARCHAR(50),
    staff_street VARCHAR(100),
    staff_state VARCHAR(50),
    staff_city VARCHAR(50),
    staff_country VARCHAR(100),
    staff_contact_number VARCHAR(20)
);

INSERT INTO gold.dim_staff (
	staff_id, 
	staff_creation_date_key, 
	staff_name, 
	staff_job_level,
	staff_street, 
	staff_state, 
	staff_city, 
	staff_country, 
	staff_contact_number
)
SELECT 
    s.staff_id::INT,
    d.date_key,
    s.name,
    s.job_level,
    s.street,
    s.state,
    s.city,
    s.country,
    s.contact_number
FROM silver.staff_data s
LEFT JOIN gold.dim_date d ON s.creation_date = d.full_date;


-- ============================================================
-- 6. DIM_USER
-- ============================================================
DROP TABLE IF EXISTS gold.dim_user CASCADE;
CREATE TABLE gold.dim_user (
    user_key SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    user_creation_date_key INT,
    user_name VARCHAR(100),
    user_street VARCHAR(100),
    user_state VARCHAR(50),
    user_city VARCHAR(50),
    user_country VARCHAR(100),
    user_birthdate_key INT,
    user_gender VARCHAR(20),
    user_device_address VARCHAR(100),
    user_type VARCHAR(50),
    user_credit_card_number BIGINT,
    user_issuing_bank VARCHAR(50),
    user_job_title VARCHAR(100),
    user_job_level VARCHAR(50)
);

INSERT INTO gold.dim_user (
    user_id, 
	user_creation_date_key, 
	user_name, 
	user_street, 
	user_state, 
	user_city, 
	user_country, 
    user_birthdate_key, 
	user_gender, 
	user_device_address, 
	user_type, 
    user_credit_card_number, 
	user_issuing_bank, 
	user_job_title, 
	user_job_level
)
SELECT
    u.user_id::INT,
    d1.date_key,
    u.name,
    u.street,
    u.state,
    u.city,
    u.country,
    d2.date_key,
    u.gender,
    u.device_address,
    u.user_type,
    cc.credit_card_number,
    cc.issuing_bank,
    j.job_title,
    j.job_level
FROM silver.user_data u
LEFT JOIN silver.user_credit_card cc ON cc.user_id = u.user_id
LEFT JOIN silver.user_job j ON J.user_id = j.user_id
LEFT JOIN gold.dim_date d1 ON u.creation_date = d1.full_date
LEFT JOIN gold.dim_date d2 ON u.birthdate = d2.full_date;


-- ============================================================
-- 7. FACT_ORDER
-- ============================================================
DROP TABLE IF EXISTS gold.fact_order CASCADE;
CREATE TABLE gold.fact_order (
    order_key SERIAL PRIMARY KEY,
    order_id VARCHAR(50),
    user_key INT,
    campaign_key INT,
    merchant_key INT,
    staff_key INT,
    transaction_date_key INT,
    estimated_arrival VARCHAR(50),
    availed BOOLEAN,
    delay_in_days INT
);

INSERT INTO gold.fact_order (
	order_id,
	user_key,
	campaign_key,
	merchant_key,
	staff_key,
	transaction_date_key,
	estimated_arrival,
	availed,
	delay_in_days
)
SELECT
	o.order_id,
	u.user_key,
	c.campaign_key,
	m.merchant_key,
	s.staff_key,
	d.date_key AS transaction_date_key,
	o.estimated_arrival,
	CASE 
        WHEN t.availed = 1 THEN TRUE
        ELSE FALSE
    END AS availed,
    od.delay_in_days
FROM silver.order_data o
LEFT JOIN silver.transactional_campaign_data t ON o.order_id = t.order_id
LEFT JOIN gold.dim_user u ON o.user_id::INT = u.user_id::INT
LEFT JOIN gold.dim_campaign c ON t.campaign_id::INT = c.campaign_id
LEFT JOIN silver.order_with_merchant_data md ON o.order_id = md.order_id
LEFT JOIN gold.dim_merchant m ON md.merchant_id::INT = m.merchant_id
LEFT JOIN gold.dim_staff s ON md.staff_id::INT = s.staff_id
LEFT JOIN gold.dim_date d ON o.transaction_date = d.full_date
LEFT JOIN silver.order_delays od ON o.order_id = od.order_id;


-- ============================================================
-- 8. FACT_LINE_ITEM
-- ============================================================
DROP TABLE IF EXISTS gold.fact_line_item CASCADE;
CREATE TABLE gold.fact_line_item (
    line_item_key SERIAL PRIMARY KEY,
    order_key INT,
    product_key INT,
    quantity INT,
    price DECIMAL(10,2)
);

INSERT INTO gold.fact_line_item (
	order_key, 
	product_key, 
	quantity, 
	price)
SELECT 
    fo.order_key,
    dp.product_key,
    lip.quantity::INT,
    lip.price
FROM silver.line_item_data_prices lip
LEFT JOIN silver.line_item_data_products liprod ON lip.order_id = liprod.order_id 
JOIN gold.fact_order fo ON lip.order_id = fo.order_id
LEFT JOIN gold.dim_product dp ON liprod.product_id::INT = dp.product_id::INT
LIMIT 100 OFFSET 0;