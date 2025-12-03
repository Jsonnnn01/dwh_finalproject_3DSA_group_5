CREATE OR REPLACE VIEW silver.v_dq_summary AS
SELECT
    table_name,
    rule_name,
    issue_type,
    severity,
    total_rows,
    failed_rows,
    CASE
        WHEN total_rows = 0 THEN 0
        ELSE failed_rows::numeric / total_rows::numeric
    END AS failed_ratio,
    created_at
FROM silver.dq_summary;

CREATE OR REPLACE VIEW silver.v_dq_invalid_counts AS
SELECT 'campaign_data_raw'::text AS table_name, COUNT(*) AS invalid_rows
FROM silver.invalid_campaign_data_raw
UNION ALL
SELECT 'merchant_data_raw', COUNT(*) FROM silver.invalid_merchant_data_raw
UNION ALL
SELECT 'staff_data_raw', COUNT(*) FROM silver.invalid_staff_data_raw
UNION ALL
SELECT 'user_data_raw', COUNT(*) FROM silver.invalid_user_data_raw
UNION ALL
SELECT 'user_job_raw', COUNT(*) FROM silver.invalid_user_job_raw
UNION ALL
SELECT 'user_credit_card_raw', COUNT(*) FROM silver.invalid_user_credit_card_raw
UNION ALL
SELECT 'product_list_raw', COUNT(*) FROM silver.invalid_product_list_raw
UNION ALL
SELECT 'order_delays_raw', COUNT(*) FROM silver.invalid_order_delays_raw
UNION ALL
SELECT 'order_data_20200101_20200701_raw', COUNT(*) FROM silver.invalid_order_data_20200101_20200701_raw
UNION ALL
SELECT 'order_data_20200701_20211001_raw', COUNT(*) FROM silver.invalid_order_data_20200701_20211001_raw
UNION ALL
SELECT 'order_data_20211001_20220101_raw', COUNT(*) FROM silver.invalid_order_data_20211001_20220101_raw
UNION ALL
SELECT 'order_data_20220101_20221201_raw', COUNT(*) FROM silver.invalid_order_data_20220101_20221201_raw
UNION ALL
SELECT 'order_data_20221201_20230601_raw', COUNT(*) FROM silver.invalid_order_data_20221201_20230601_raw
UNION ALL
SELECT 'order_data_20230601_20240101_raw', COUNT(*) FROM silver.invalid_order_data_20230601_20240101_raw
UNION ALL
SELECT 'order_with_merchant_data1_raw', COUNT(*) FROM silver.invalid_order_with_merchant_data1_raw
UNION ALL
SELECT 'order_with_merchant_data2_raw', COUNT(*) FROM silver.invalid_order_with_merchant_data2_raw
UNION ALL
SELECT 'order_with_merchant_data3_raw', COUNT(*) FROM silver.invalid_order_with_merchant_data3_raw
UNION ALL
SELECT 'line_item_prices1_raw', COUNT(*) FROM silver.invalid_line_item_prices1_raw
UNION ALL
SELECT 'line_item_prices2_raw', COUNT(*) FROM silver.invalid_line_item_prices2_raw
UNION ALL
SELECT 'line_item_prices3_raw', COUNT(*) FROM silver.invalid_line_item_prices3_raw
UNION ALL
SELECT 'line_item_products1_raw', COUNT(*) FROM silver.invalid_line_item_products1_raw
UNION ALL
SELECT 'line_item_products2_raw', COUNT(*) FROM silver.invalid_line_item_products2_raw
UNION ALL
SELECT 'line_item_products3_raw', COUNT(*) FROM silver.invalid_line_item_products3_raw;