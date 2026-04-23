"""
E-Commerce Sales EDA
Author: Gunjan
Description: Exploratory Data Analysis on e-commerce sales data
             to uncover trends, top products, and customer behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#6c63ff','#ff6584','#43aa8b','#f59f00','#4ecdc4','#e63946']

# ── Generate Dataset ───────────────────────────────────────
def generate_ecommerce_data(n=5000):
    np.random.seed(42)
    categories  = ['Electronics','Clothing','Home & Kitchen','Sports','Books','Beauty']
    cities      = ['Delhi','Mumbai','Bangalore','Hyderabad','Chennai','Pune','Kolkata']
    payment     = ['Credit Card','UPI','Debit Card','Net Banking','COD']
    cat_prices  = {'Electronics':(5000,80000),'Clothing':(500,5000),
                   'Home & Kitchen':(1000,15000),'Sports':(800,8000),
                   'Books':(200,1500),'Beauty':(300,3000)}

    dates = pd.date_range('2023-01-01','2023-12-31', periods=n)
    cat   = np.random.choice(categories, n, p=[0.25,0.20,0.18,0.15,0.12,0.10])

    prices, discounts, ratings = [], [], []
    for c in cat:
        lo, hi = cat_prices[c]
        prices.append(np.random.randint(lo, hi))
        discounts.append(np.random.choice([0,5,10,15,20,25,30], p=[0.2,0.15,0.2,0.15,0.15,0.1,0.05]))
        ratings.append(round(np.random.uniform(2.5, 5.0), 1))

    prices    = np.array(prices)
    discounts = np.array(discounts)
    final_prices = prices * (1 - discounts/100)

    df = pd.DataFrame({
        'order_id':    [f'ORD{i:05d}' for i in range(1, n+1)],
        'date':        dates,
        'category':    cat,
        'city':        np.random.choice(cities, n),
        'payment':     np.random.choice(payment, n, p=[0.30,0.35,0.15,0.10,0.10]),
        'quantity':    np.random.randint(1, 6, n),
        'unit_price':  prices,
        'discount_pct':discounts,
        'final_price': final_prices.astype(int),
        'rating':      ratings,
        'returned':    np.random.choice([0,1], n, p=[0.92,0.08]),
    })
    df['revenue']     = df['final_price'] * df['quantity']
    df['profit']      = (df['revenue'] * np.random.uniform(0.15, 0.35, n)).astype(int)
    df['month']       = df['date'].dt.month
    df['month_name']  = df['date'].dt.strftime('%b')
    df['weekday']     = df['date'].dt.day_name()
    df['quarter']     = df['date'].dt.quarter
    return df


def run_eda(df):
    import os; os.makedirs('outputs', exist_ok=True)
    print(f"Dataset: {df.shape[0]:,} orders | Revenue: ₹{df['revenue'].sum()/1e7:.2f} Cr")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('E-Commerce Sales EDA Dashboard 2023', fontsize=17, fontweight='bold')

    # 1. Revenue by Category
    cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
    axes[0,0].bar(cat_rev.index, cat_rev.values/1e6, color=COLORS)
    axes[0,0].set_title('Revenue by Category (₹ Millions)', fontweight='bold')
    axes[0,0].set_ylabel('Revenue (Millions)')
    axes[0,0].tick_params(axis='x', rotation=20)

    # 2. Monthly Revenue Trend
    monthly = df.groupby('month')['revenue'].sum()
    axes[0,1].plot(monthly.index, monthly.values/1e6, 'o-', color='#6c63ff', linewidth=2.5, markersize=7)
    axes[0,1].fill_between(monthly.index, monthly.values/1e6, alpha=0.15, color='#6c63ff')
    axes[0,1].set_title('Monthly Revenue Trend', fontweight='bold')
    axes[0,1].set_xlabel('Month'); axes[0,1].set_ylabel('Revenue (Millions)')
    axes[0,1].set_xticks(range(1,13))
    axes[0,1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

    # 3. Payment Method Distribution
    pay = df['payment'].value_counts()
    axes[0,2].pie(pay.values, labels=pay.index, colors=COLORS, autopct='%1.1f%%',
                  startangle=90, wedgeprops={'edgecolor':'white','linewidth':1.5})
    axes[0,2].set_title('Payment Method Distribution', fontweight='bold')

    # 4. Top Cities by Revenue
    city_rev = df.groupby('city')['revenue'].sum().sort_values(ascending=True)
    axes[1,0].barh(city_rev.index, city_rev.values/1e6, color='#43aa8b')
    axes[1,0].set_title('Revenue by City (₹ Millions)', fontweight='bold')
    axes[1,0].set_xlabel('Revenue (Millions)')

    # 5. Rating Distribution
    axes[1,1].hist(df['rating'], bins=25, color='#ff6584', edgecolor='white', alpha=0.85)
    axes[1,1].axvline(df['rating'].mean(), color='#6c63ff', linestyle='--',
                      linewidth=2, label=f'Mean: {df["rating"].mean():.2f}')
    axes[1,1].set_title('Customer Rating Distribution', fontweight='bold')
    axes[1,1].set_xlabel('Rating'); axes[1,1].legend()

    # 6. Revenue by Quarter & Category (heatmap)
    pivot = df.pivot_table(values='revenue', index='category', columns='quarter',
                           aggfunc='sum') / 1e6
    sns.heatmap(pivot.round(1), ax=axes[1,2], annot=True, fmt='.1f',
                cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Revenue (M)'})
    axes[1,2].set_title('Revenue Heatmap: Category × Quarter', fontweight='bold')
    axes[1,2].set_xlabel('Quarter')

    plt.tight_layout()
    plt.savefig('outputs/eda_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

    df.to_csv('outputs/ecommerce_processed.csv', index=False)
    print("Saved: outputs/eda_dashboard.png | outputs/ecommerce_processed.csv")

    print("\n── Key Insights ──────────────────────────────")
    print(f"  Top Category   : {cat_rev.index[0]} (₹{cat_rev.iloc[0]/1e6:.1f}M)")
    print(f"  Avg Order Value: ₹{df['revenue'].mean():,.0f}")
    print(f"  Return Rate    : {df['returned'].mean()*100:.1f}%")
    print(f"  Top City       : {df.groupby('city')['revenue'].sum().idxmax()}")
    print(f"  Best Month     : {df.groupby('month')['revenue'].sum().idxmax()}")
    print(f"  Avg Rating     : {df['rating'].mean():.2f} / 5.0")


if __name__ == '__main__':
    df = generate_ecommerce_data()
    run_eda(df)
