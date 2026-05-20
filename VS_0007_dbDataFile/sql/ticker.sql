-- SQLite
-- drop table ticker;
create table ticker (
    id integer primary key autoincrement,
    datetime1 datetime, 
    date1 date,
    time1 time,
    hign real,
    low real,
    close real,
    open  real,
    volume integer,
    symbol text
    -- price real not null,
    --timestamp datetime default current_timestamp
);