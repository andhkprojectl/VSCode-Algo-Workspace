-- 1. Create the database (schema)
CREATE DATABASE IBTradingDb;

-- 2. Create the user and set a password
-- Replace 'your_password' with a strong password
CREATE USER 'ibUser1'@'localhost' IDENTIFIED BY 'abcd_9538';

-- 3. Grant full privileges to this user for the new database
GRANT ALL PRIVILEGES ON IBTradingDb.* TO 'ibUser1'@'localhost';

-- 4. Apply the changes immediately
FLUSH PRIVILEGES;