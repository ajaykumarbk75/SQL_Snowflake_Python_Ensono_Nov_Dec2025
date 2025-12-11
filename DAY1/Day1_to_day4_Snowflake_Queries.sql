// To Create a database 
CREATE DATABASE ajay_dtabase211;

// This is Comment is Snowflake SQL 
//1. Create a Simple Table
 
CREATE OR REPLACE TABLE CUSTOMER (
    CUSTOMER_ID INT,
    NAME STRING,
    CITY STRING,
    AGE INT
);

// Insert Data into Table
INSERT INTO CUSTOMER VALUES
(1, 'Rahul', 'Mumbai', 29),
(2, 'Sneha', 'Delhi', 30),
(3, 'John', 'Bangalore', 45),
(1, 'Meera', 'Ahmedabad', 35)

// QUERY / VIEW data from the table 

SELECT * FROM CUSTOMER; 


// Create table called employee & Insert data into it

CREATE TABLE employees (

   emp_id INT AUTOINCREMENT,
   first_name STRING,
   last_name  STRING,
   email STRING,
   department_ID  INT,
   hire_date DATE,
   salary NUMBER(10,2)   // Salary is decimal hene 2 Decimal points selecting 10,000.00 USD
);
   
INSERT INTO  employees(first_name,last_name,email,department_ID, hire_date,salary) VALUES
('John', 'Brown', 'john.brown@abccorp.com', 101, '2025-03-10', 65000),
('Sara', 'Miller', 'Sara.Miller@abccorp.com', 102, '2025-02-10', 75000),
('David', 'Wilson', 'David.Wilson@abccorp.com', 101, '2024-08-21', 75000),
('Emma', 'Davis', 'Emma.Davis@abccorp.com', 103, '2025-05-10', 75000);

SELECT * FROM employees;



// Usage of WHERE Cluase - CONDITION

SELECT * FROM employees
WHERE department_id = 101;

// Depertment-Id = 102

// Usage of WHERE Cluase - CONDITION
SELECT * FROM employees
WHERE department_id = 102;

// Usage of WHERE Cluase - CONDITION
SELECT * FROM employees
WHERE department_id = 103;

// With String values

SELECT * FROM employees
WHERE first_name = 'John';

// List employees from Departments 101, 103
SELECT * FROM employees
WHERE department_id IN (101, 103);


// WHERE clause with NOT IN Operator 

SELECT * FROM employees
WHERE department_id NOT IN (101, 103);


// Group By Commands - Group the certain records & Display the Results 

SELECT * FROM employees;

SELECT 
   department_id,
   COUNT(*) AS Total_employees_Count_In_Department
FROM employees
GROUP BY department_id;

// Show total Employees & Average SALARY per department
SELECT 
    department_id,
    COUNT(*) AS Emp_Count,    // COUNT -> Function counts total people
    AVG(salary) AS avg_salary // AVG -> Function Does average 
FROM employees
GROUP BY department_id;

// Rounding to 2 decimal places 

SELECT 
    department_id,
    COUNT(*) AS Emp_Count,              -- Total people 
    ROUND(AVG(salary), 2) AS avg_salary -- Round to 2 decimal places 
FROM employees
GROUP BY department_id;

// Maximum & Minimum Salary per department

SELECT 
    department_id,
    MAX(salary) AS max_salary,  // Max Function - takes Max value 
    MIN(salary) AS min_salary    // Min Function - takes Max value 
    
FROM employees
GROUP BY department_id;

// ORDER BY Queries - 

//SELECT * FROM employees

SELECT * FROM employees
ORDER BY salary DESC;

// Choose Department_Id column

SELECT * FROM employees
ORDER BY department_id, salary DESC;


// Salary ASCENDING 
SELECT * FROM employees
ORDER BY department_id, salary ASC;

// department_id DESC & salary ASC 
SELECT * FROM employees
ORDER BY department_id DESC, salary ASC;


// Group By Hire Year 
SELECT 
  YEAR(hire_date) AS hire_year,  // Year Function - extracts YEAR values from given Column Record 
  COUNT(*) AS total_hired
FROM employees
GROUP BY hire_year
ORDER BY hire_year;


// Salary - Hire date 

SELECT 
    department_id,
    SUM(CASE WHEN YEAR(hire_date) = 2024 THEN salary ELSE 0 END) AS total_salary_2024,
    SUM(CASE WHEN YEAR(hire_date) = 2025 THEN salary ELSE 0 END) AS total_salary_2025
FROM employees
GROUP BY department_id
ORDER BY department_id;


// Customer & Products Tables

SELECT * FROM CUSTOMER;   


CREATE OR REPLACE TABLE products (
    product_id INT autoincrement,
    product_name STRING,
    price NUMBER(10, 2)

);

//Insert data into products

INSERT INTO products(product_name, price )  VALUES
('Laptop', 50000),
('Mouse', 500),
('Keyboard', 1500),
('Monitor', 12000);

SELECT * FROM products;


// Create table - ORDERS

CREATE OR REPLACE TABLE orders (
    order_id INT autoincrement,
    customer_id INT,
    product_id INT,
    quantity INT,
    order_date DATE DEFAULT CURRENT_DATE()
);

INSERT INTO orders (customer_id,product_id , quantity )  VALUES 
(1, 1, 1),
(1, 2, 2),
(2, 4, 1),
(3, 3, 1),
(4, 2, 3);

SELECT * FROM orders;


// VIew current date & Time

SELECT CURRENT_TIME() AS current_time;

SELECT * FROM CUSTOMER;  

SELECT * FROM orders;

SELECT * FROM products;  


// INNER Join -> Happens only we have Matching records 

// Customers with ordered - products 

SELECT 
  o.order_id,
  c.name AS customer_name,
  p.product_name,
  o.quantity,
  p.price,
  (o.quantity * p.price) AS Total_Amount
FROM orders o
INNER JOIN customer c 
   ON o.customer_id = c.customer_id
INNER JOIN products p
   ON o.product_id = p.product_id;
  

// Left Join - happens if match happens else it will return NULL

SELECT 
   c.name AS customer_name,
   o.order_id,
   o.quantity
FROM customer c
LEFT JOIN orders o
   ON c.customer_id = o.customer_id 
  

// Right Join - pefrence - Right table 

SELECT 
   c.name AS customer_name,
     o.order_id,
     o.quantity
FROM customer c
RIGHT JOIN orders o
   ON c.customer_id = o.customer_id 


// Subqueries - Conditions & JOINS 
// Find customers who spent more than 20,000

SELECT name
FROM customer
WHERE customer_id IN (
    SELECT customer_id
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    GROUP BY customer_id
    HAVING SUM(o.quantity * p.price) > 20000
    ); 


// ==== CTE - Common Table Expression 
// Syntax for CTE :

WITH cte_name  AS (
        SEELCT .....

)

SELECT * FROM cte_name;

// CTE - Employees with Salary above the department average:
//SELECT * FROM employees;

WITH dept_avg AS (
    SELECT 
       department_id,
       AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT 
    e.first_name,
    e.last_name,
    e.department_id,
    e.salary,
    d.avg_salary     /// I am referring d -> dept_avg CTE 
FROM employees e
JOIN dept_avg d
   ON e.department_id = d.department_id
WHERE e.salary > d.avg_salary;


// CTE - Calculate the Total Order Amount per Order
//SELECT * FROM orders;
//SELECT * FROM products; 

WITH  order_amount  AS (    // CTE always created in the runtime - it is faster
    SELECT 
    o.order_id,
    o.customer_id,
    o.product_id,
    o.quantity,
    p.price,
    (o.quantity * p.price ) AS Total_Amount
    FROM orders o
    JOIN products p
      ON o.product_id = p.product_id  
) 

SELECT * FROM order_amount;


//  CTE - Employees hired in 2025 Only 
//SELECT * FROM employees;
WITH hired_2025 AS (
    SELECT * FROM employees
    WHERE YEAR(hire_date) = 2025
)
SELECT * FROM hired_2025;


// CTE - RANK() Function 
WITH salary_rank AS (
    SELECT 
        first_name,
        last_name,
        department_id, 
        salary,
        RANK() OVER(PARTITION BY department_id ORDER BY salary DESC) AS salary_rank // Department 
        FROM employees
)
SELECT * FROM salary_rank;

// CTE - Most Expensive Product Ordered
SELECT * FROM orders;
SELECT * FROM products;

WITH product_cost  AS (
    SELECT 
      o.order_id,
      p.product_name,
      p.price,
      RANk() OVER (ORDER BY p.price DESC) AS price_rank
      FROM orders o
      JOIN products p ON o.product_id = p.product_id
) 
SELECT * FROM product_cost
WHERE price_rank = 1;

// CTE - with MAX() function 

WITH product_cost AS (
    SELECT 
      p.product_name,
      p.price
    FROM  orders o
    JOIN products p ON o.product_id = p.product_id

)
SELECT * FROM product_cost 
WHERE price = (SELECT MAX(price) FROM product_cost);

// ROW Number(): Get the latest order per customer
WITH  ranked AS (
    SELECT 
        customer_id,
        order_id,
        order_date,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC ) AS rn
        FROM orders
) 

SELECT * FROM ranked
WHERE rn = 1;

// LEAD() query - Next Order date for Each Customer
SELECT 
   customer_id,
   order_id,
   order_date,
   LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date
FROM orders
ORDER BY customer_id, order_date;

// LEAD()- Next Product ordered by Customer 
//SELECT * FROM products;
SELECT 
 customer_id,
 order_id,
 product_id,
 LEAD(product_id) OVER ( PARTITION BY customer_id  ORDER BY order_id) AS next_product_id

 FROM ORDERS
 ORDER BY customer_id,order_id;

//LAG() -  Previous Salary in the same Department
SELECT 
    first_name,
    last_name,
    department_id,
    salary,
    LAG(salary) OVER (PARTITION BY department_id  ORDER BY hire_date ) AS previous_salary
FROM employees
ORDER BY department_id, hire_date;

// LEAD() - with case statement -> to check if customer returns or not 
//SELECT * FROM orders;
SELECT 
   customer_id,
   order_id,
   CASE 
       WHEN LEAD(customer_id) OVER ( PARTITION BY customer_id ORDER BY order_id ) IS NULL
       THEN 'No NEXT Order'
       ELSE 'Has Next ORDER'
    END AS customer_return_status
FROM orders
ORDER BY customer_id,  order_id;


// SQL performance & efficiency :
SELECT * FROM orders;  -> Whole table will be scanned - more time 
SELECT order_id, customer_id FROM orders;  -> this is recommended



// Churn dataset -> data which tells su that - Customers are stopping our services / Products etc 
// Churn Dataset - Telecom dataset 

CREATE OR REPLACE DATABASE telecom_churn_db;
USE DATABASE telecom_churn_db;

// Customer table   creation 

CREATE OR REPLACE TABLE customers (
  customer_id INT,
  name STRING,
  city STRING,
  age INT,
  tenure_months INT,
  is_active STRING
);


// Plan table creation 
CREATE OR REPLACE TABLE plans (
  plan_id INT,
  plan_name STRING,
  monthly_charges NUMBER(10, 2) 
);

// Subscription Table 

CREATE OR REPLACE TABLE subscriptions (
   customer_id INT,
   plan_id INT,
   start_date DATE,
   end_date DATE
   );

  --  //  customer_id INT,
  -- name STRING,
  -- city STRING,
  -- age INT,
  -- tenure_months INT,
  -- is_active STRING 
INSERT INTO customers VALUES 
    (1, 'Rahul', 'Mumbai',32,12, 'YES'),
    (2, 'Suhas', 'Chennai',35,6, 'No'),
    (3, 'John', 'Bangalore',45,36, 'YES'),
    (4, 'Kiran', 'Pune',40,2, 'No'),
    (5, 'Mahesh', 'Hyderabad',28,12, 'YES')
;

SELECT * FROm customers;

INSERT INTO plans VALUES 
(101, 'Basic', 299),
(102, 'Premium', 499),
(103, 'Enterprise', 999);

INSERT INTO subscriptions VALUES 
(1, 101, '2025-01-01', NULL),
(2, 102, '2025-02-10', '2025-08-10'),
(3, 103, '2025-05-01', NULL),
(4, 104, '2025-03-15', '2025-05-01');


// INNER JOIN - Active Customer with Plan details:

SELECT 
  c.customer_id,
  c.name,
  p.plan_name,
  p.monthly_charges
FROM customers c
INNER JOIN subscriptions s ON c.customer_id = s.customer_id
INNER JOIN plans p ON s.plan_id = p.plan_id
WHERE c.is_active = 'YES';

// Left JOIN - Show All customers even of they don't have a subscription 

SELECT 
   c.name,
   s.plan_id,
   p.plan_name

FROM customers c
LEFT JOIN subscriptions s ON  c.customer_id = s.customer_id
LEFT JOIN plans p ON s.plan_id = p.plan_id;

// Full Outer Join - Identify the mismatches
// It is Like A U B (A union B ) - it clarifies what are missing from both tables

SELECT * FROM customers c
FULL OUTER JOIN subscriptions s ON c.customer_id = s.customer_id;

// Identify the Churned Customers:
SELECT 
  c.name,
  c.city,
  c.tenure_months,
  CASE  
        WHEN s.end_date IS NOT NULL THEN 'Churned'
        ELSE 'Active'
  END AS churn_status

FROM customers c
LEFT JOIN subscriptions s ON c.customer_id = s.customer_id;



SELECT * FROm customers;
SELECT * FROm subscriptions;
SELECT * FROm plans; 


// Data Modeling & analytics 
CREATE OR REPLACE TABLE fact_subscriptions AS 
SELECT 
    s.customer_id,
    s.plan_id,
    s.start_date,
    s.end_date,
    p.monthly_charges,
    DATEDIFF('day', s.start_date, NVL(s.end_date, CURRENT_DATE)) AS total_days,
    CASE WHEN s.end_date IS NULL THEN 'Active' ELSE 'Churned' END AS status
FROM subscriptions s
JOIN plans p ON s.plan_id = p.plan_id;
    

SELECT * FROM fact_subscriptions;


// Mini - Project - DIWALI_SALES_TABLE1 - Apply analytics 

SELECT * FROM AJAY_DTABASE211.PUBLIC.DIWALI_SALES_TABLE1;

SELECT * FROM DIWALI_SALES_TABLE1;
       
// Basic Exploration - Total Number of ROWS 

SELECT COUNT(*)  AS total_rows FROM DIWALI_SALES_TABLE1;

// Missing Amount records

SELECT COUNT(*)  missing_amount
FROM DIWALI_SALES_TABLE1
WHERE Amount is NULL OR Amount = 0;

// TOTAL Amount FROM All Customer 
SELECT SUM(Amount) AS Total_Revenue
FROm DIWALI_SALES_TABLE1;

// ORDERS & Revenue BY GENDER
SELECT 
    Gender,
    COUNT(*) AS Total_Customer,
    SUM(ORDERS) AS total_orders,
    SUM(Amount) AS revenue

FROM DIWALI_SALES_TABLE1
GROUP BY Gender
ORDER BY revenue DESC;


// StateWise revenue - Top 10 States
SELECT STATE,
  SUM(Amount) AS Total_Revenue
FROM DIWALI_SALES_TABLE1
GROUP BY State
ORDER BY Total_Revenue DESC
LIMIT 10;    // Top 10 States with highest Revenue
  

//Zone wise Customer Distribution 
SELECT 
   Zone,
   COUNT(User_id) AS customers
FROM DIWALI_SALES_TABLE1
GROUP BY Zone;

// Purchase Behaviour - Behavioural Analytics
SELECT 
   Age_Group,
   SUM(Orders) AS total_orders,
   SUM(Amount) AS total_Sales
FROm DIWALI_SALES_TABLE1
GROUP By AGE_GROUP
ORDER By total_Sales DESC;


