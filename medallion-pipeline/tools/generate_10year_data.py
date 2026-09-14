from pathlib import Path
import random
import pandas as pd
from datetime import datetime

random.seed(42)
landing = Path('data/landing')
landing.mkdir(parents=True, exist_ok=True)

products = pd.DataFrame([
    {'product_id':'P01','product_name':'Wireless Headphones','category':'Electronics','standard_price':80.0},
    {'product_id':'P02','product_name':'Office Chair','category':'Furniture','standard_price':120.0},
    {'product_id':'P03','product_name':'Mechanical Keyboard','category':'Accessories','standard_price':60.0},
    {'product_id':'P04','product_name':'Smart Watch','category':'Electronics','standard_price':150.0},
    {'product_id':'P05','product_name':'Running Shoes','category':'Footwear','standard_price':70.0},
    {'product_id':'P06','product_name':'Air Fryer','category':'Appliances','standard_price':90.0},
    {'product_id':'P07','product_name':'Backpack','category':'Lifestyle','standard_price':50.0},
    {'product_id':'P08','product_name':'Coffee Maker','category':'Appliances','standard_price':75.0},
    {'product_id':'P09','product_name':'Cotton T-Shirt','category':'Apparel','standard_price':20.0},
    {'product_id':'P10','product_name':'Jeans','category':'Apparel','standard_price':45.0},
    {'product_id':'P11','product_name':'LED Desk Lamp','category':'Electronics','standard_price':35.0},
    {'product_id':'P12','product_name':'Bluetooth Speaker','category':'Electronics','standard_price':50.0},
    {'product_id':'P13','product_name':'Winter Jacket','category':'Apparel','standard_price':110.0},
    {'product_id':'P14','product_name':'Tablet Stand','category':'Accessories','standard_price':25.0},
    {'product_id':'P15','product_name':'USB-C Cable','category':'Accessories','standard_price':15.0},
    {'product_id':'P16','product_name':'Notebook Set','category':'Office','standard_price':12.0},
    {'product_id':'P17','product_name':'Yoga Mat','category':'Fitness','standard_price':30.0},
    {'product_id':'P18','product_name':'Water Bottle','category':'Lifestyle','standard_price':18.0},
    {'product_id':'P19','product_name':'Desk Organizer','category':'Office','standard_price':22.0},
    {'product_id':'P20','product_name':'Phone Case','category':'Accessories','standard_price':16.0},
])

stores = pd.DataFrame([
    {'store_id':'S01','store_name':'New York Flagship','region':'Northeast','city':'New York','state':'NY'},
    {'store_id':'S02','store_name':'Chicago Urban Store','region':'Midwest','city':'Chicago','state':'IL'},
    {'store_id':'S03','store_name':'Austin Outlet','region':'South','city':'Austin','state':'TX'},
    {'store_id':'S04','store_name':'Seattle Market','region':'West','city':'Seattle','state':'WA'},
])

products.to_csv(landing / 'products.csv', index=False)
stores.to_csv(landing / 'stores.csv', index=False)

rows = []
for year in range(2016, 2026):
    for month in range(1, 13):
        for i in range(50):
            product = random.choice(products.to_dict('records'))
            store = random.choice(stores.to_dict('records'))
            quantity = random.randint(1, 7)
            unit_price = float(product['standard_price'])
            total_amount = round(quantity * unit_price * random.uniform(0.95, 1.40), 2)
            rows.append({
                'transaction_id': f'TXN-{year}-{month:02d}-{i+1:04d}',
                'transaction_date': datetime(year, month, random.randint(1, 28)).strftime('%Y-%m-%d'),
                'store_id': store['store_id'],
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'category': product['category'],
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total_amount,
                'customer_segment': random.choice(['Retail','Corporate','Online','VIP']),
                'payment_method': random.choice(['Card','Cash','UPI','Wallet']),
            })

sales = pd.DataFrame(rows)
sales.to_csv(landing / 'sales_data.csv', index=False)
print(f'created {len(sales)} sales rows across 2016-2025')
