from datetime import date, timedelta
print("# Date")
start_date = date(2021, 1, 1)
end_date = date(2022, 12, 31)

current_date = start_date
while current_date <= end_date:
    print(current_date.strftime("%Y-%m-%d"))
    current_date += timedelta(days=1)