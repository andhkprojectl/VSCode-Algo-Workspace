-- SQLite
create table ticker (
    id integer primary key autoincrement,
    symbol text not null,
    price real not null,
    timestamp datetime default current_timestamp
);