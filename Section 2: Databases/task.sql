Task 1:
select id,sum(Amount) as Total_spending from (select t.Amount,c.name,c.phone,c.address,c.id from tansaction t join customer c on t.Customer_id = c.id) group by id

Task 2:
select * from(select count(id) as sales_Number,Manufacture(select t.id,p.Manufacture from tansaction t join product p on t.product_id = p.id and EXTRACT(
    MONTH FROM t.datetime
    ) == EXTRACT(
    MONTH FROM current_timestamp
    ) ) Group by Manufacture) order by sales_Number desc limit 3