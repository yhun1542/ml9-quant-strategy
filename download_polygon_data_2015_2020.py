"""
Download historical price data for 2015-2020 using Polygon Flat Files (S3)
"""

import boto3
import pandas as pd
from pathlib import Path
from io import BytesIO
import gzip
from datetime import datetime, timedelta

# Polygon S3 credentials
AWS_ACCESS_KEY_ID = "f0bc904a-9d5c-476b-af56-2cb4a2455a3e"
AWS_SECRET_ACCESS_KEY = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"
S3_ENDPOINT = "https://files.polygon.io"
BUCKET = "flatfiles"

# 30 tickers
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]

# Date range
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2020, 12, 31)

print("="*100)
print("DOWNLOADING POLYGON FLAT FILES (2015-2020)")
print("="*100)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=S3_ENDPOINT
)

print(f"\n✅ S3 client initialized")
print(f"Endpoint: {S3_ENDPOINT}")
print(f"Bucket: {BUCKET}")

# Download data for each ticker
all_data = {}

for ticker in TICKERS:
    print(f"\nProcessing {ticker}...")
    
    ticker_data = []
    
    # Iterate through years
    for year in range(START_DATE.year, END_DATE.year + 1):
        # Polygon flat file path: us_stocks_sip/day_aggs_v1/2015/01/2015-01-02.csv.gz
        for month in range(1, 13):
            # Skip months outside date range
            if year == START_DATE.year and month < START_DATE.month:
                continue
            if year == END_DATE.year and month > END_DATE.month:
                continue
            
            # Try to download monthly file
            # Polygon stores daily files, so we'll try a few dates
            for day in [1, 2, 3, 15]:  # Sample a few days
                try:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    s3_key = f"us_stocks_sip/day_aggs_v1/{year}/{month:02d}/{date_str}.csv.gz"
                    
                    # Download file
                    response = s3_client.get_object(Bucket=BUCKET, Key=s3_key)
                    
                    # Decompress and read CSV
                    with gzip.GzipFile(fileobj=BytesIO(response['Body'].read())) as gz:
                        df = pd.read_csv(gz)
                    
                    # Filter for this ticker
                    ticker_df = df[df['ticker'] == ticker].copy()
                    
                    if not ticker_df.empty:
                        ticker_data.append(ticker_df)
                        print(f"  {date_str}: {len(ticker_df)} rows")
                        break  # Found data for this month, move to next
                
                except Exception as e:
                    # File might not exist for this date
                    continue
    
    if ticker_data:
        # Combine all data for this ticker
        ticker_df = pd.concat(ticker_data, ignore_index=True)
        ticker_df = ticker_df.sort_values('date')
        ticker_df = ticker_df.drop_duplicates(subset=['date'])
        
        # Use close price (adjusted)
        all_data[ticker] = ticker_df.set_index('date')['close']
        print(f"✅ {ticker}: {len(all_data[ticker])} days")
    else:
        print(f"⚠️ {ticker}: No data found")

# Create DataFrame
if all_data:
    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = 'date'
    df = df.reset_index()
    
    # Save to CSV
    output_path = Path("/home/ubuntu/quant-ensemble-strategy/data/price_data_2015_2020_polygon.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {len(df.columns) - 1}")
else:
    print("\n❌ No data downloaded")

print("\n" + "="*100)
print("DOWNLOAD COMPLETE")
print("="*100)
