import yfinance as yf

results = yf.Search("NIFTY", max_results=200)

for item in results.quotes:
    print(
        item.get("symbol"),
        "|",
        item.get("shortname") or item.get("longname"),
        "|",
        item.get("quoteType")
    )