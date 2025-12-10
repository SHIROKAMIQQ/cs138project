import pandas as pd
from pylindol import PhivolcsEarthquakeInfoScraper

OUTPUT_FILE = "PHIVOLCS_202312Dataset.csv"

# Scrape December 2023
scraper = PhivolcsEarthquakeInfoScraper(month=12, year=2023, export_to_csv=False)
df = scraper.run()

# print(df.columns)
# Index(['Date - Time (Philippine Time)', 'Latitude  (ºN)', 'Longitude  (ºE)', 'Depth (km)', 'Mag', 'Location'], dtype='object')

df = df.rename(columns={
  'Date - Time  (Philippine Time)': 'date_time',
  'Latitude  (ºN)': 'latitude',
  'Longitude  (ºE)': 'longitude',
  'Mag': 'mag'
})
print(df.columns)
df['date_time'] = pd.to_datetime(df['date_time'], format="%d %B %Y - %I:%M %p")

# Filter for Dec 2, 2023 to Dec 12, 2023
start = pd.Timestamp("2023-12-02")
end = pd.Timestamp("2023-12-12 23:59:59")
mask = (df['date_time'] >= start) & (df['date_time'] <= end)
df_filtered = df.loc[mask]

# Save to CSV
df_filtered.to_csv(OUTPUT_FILE, index=False)