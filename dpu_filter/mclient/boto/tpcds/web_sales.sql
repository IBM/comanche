-- Query 1
SELECT ws_item_sk, ws_quantity, ws_sales_price FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451911 AND 2452640;

-- Query 2
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2452335 AND 2452456;

-- Query 3
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451180 AND 2451910;

-- Query 4
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_item_sk BETWEEN 8 AND 300) AND (ws_sold_date_sk BETWEEN 2452356 AND 2452386);

-- Query 5
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451545 AND 2451818;

-- Query 6
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451239 AND 2451604;

-- Query 7
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451636 AND 2451726;

-- Query 8
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451880 AND 2451910 
AND ws_net_profit > 1.00 AND ws_net_paid > 0.00 AND ws_quantity > 0;

-- Query 9
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2452001 AND 2452365;

-- Query 10
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_item_sk BETWEEN 19 AND 300) 
AND (ws_sold_date_sk BETWEEN 2450905 AND 2450934);

-- Query 11
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_bill_addr_sk BETWEEN 2 AND 6000) 
AND (ws_sold_date_sk BETWEEN 2450935 AND 2450965);

-- Query 12
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_bill_addr_sk BETWEEN 1 AND 6000) 
AND (ws_sold_date_sk BETWEEN 2452184 AND 2452214);

-- Query 13
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_ship_date_sk BETWEEN 2452062 AND 2452426;

-- Query 14
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_sold_time_sk BETWEEN 19072 AND 47872) 
AND (ws_sold_date_sk BETWEEN 2451911 AND 2452275) 
AND (ws_ship_mode_sk BETWEEN 2 AND 18);

-- Query 15
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2452307 AND 2452395;

-- Query 16
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_item_sk BETWEEN 1 AND 300) 
AND (ws_sold_date_sk BETWEEN 2452215 AND 2452244) 
AND (ws_sold_time_sk BETWEEN 21600 AND 71999);

-- Query 17
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451911 AND 2452640;

-- Query 18
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE (ws_item_sk BETWEEN 1 AND 300) 
AND (ws_sold_date_sk BETWEEN 2451180 AND 2451910);

-- Query 19
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_ship_hdemo_sk IS NULL;

-- Query 20
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2452133 AND 2452163;

-- Query 21
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE (ws_promo_sk BETWEEN 1 AND 500) 
AND (ws_item_sk BETWEEN 3 AND 300) 
AND (ws_sold_date_sk BETWEEN 2452491 AND 2452521);

-- Query 22
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE ws_net_profit BETWEEN 50.00 AND 300.00 
AND ws_sales_price BETWEEN 50.00 AND 200.00 
AND ws_web_page_sk IS NOT NULL 
AND ws_sold_date_sk BETWEEN 2451911 AND 2452275;

-- Query 23
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451697 AND 2452061;

-- Query 24
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451211 AND 2451575;

-- Query 25
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE (ws_web_page_sk BETWEEN 6 AND 200) 
AND (ws_ship_hdemo_sk BETWEEN 24 AND 63) 
AND (ws_sold_time_sk BETWEEN 32 AND 39);

-- Query 26
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE (ws_item_sk BETWEEN 5 AND 300) 
AND (ws_sold_date_sk BETWEEN 2451935 AND 2452025);

-- Query 27
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE (ws_ship_date_sk BETWEEN 2451270 AND 2451330) 
AND (ws_ship_addr_sk BETWEEN 27 AND 6000) 
AND (ws_web_site_sk BETWEEN 2 AND 36);

-- Query 28
SELECT ws_item_sk, ws_quantity, ws_sales_price 
FROM S3Object 
WHERE ws_sold_date_sk BETWEEN 2451180 AND 2451453;

-- Query 29
SELECT COUNT(*), SUM(ws_sales_price), SUM(ws_quantity) 
FROM S3Object 
WHERE (ws_bill_addr_sk BETWEEN 2 AND 6000) 
AND (ws_sold_date_sk BETWEEN 2451270 AND 2451299);
