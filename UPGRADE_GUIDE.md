# Obsolete Guide

The dataset-backed repricing flow described here has been removed from the active product AI agent. This file is kept only as historical reference.

# How To Swap Kaggle Dataset For Real Amazon SP-API

The repricing system uses dependency inversion:

```text
API routes -> RepricingEngine -> PriceDataProvider -> KaggleProvider or AmazonSPAPIProvider
```

This keeps business logic, schemas, and API routes unchanged when moving from demo data to real marketplace data.

## 1. Get Amazon SP-API Credentials

Create and authorize a Selling Partner API application in Seller Central. Collect the required credentials:

```env
AMAZON_CLIENT_ID=
AMAZON_CLIENT_SECRET=
AMAZON_REFRESH_TOKEN=
AMAZON_MARKETPLACE_ID=ATVPDKIKX0DR
```

## 2. Set Environment Variables

Add the credentials to the deployment environment. When all required Amazon variables are present, `app.providers.factory.get_provider()` automatically returns `AmazonSPAPIProvider`.

## 3. Implement AmazonSPAPIProvider Methods

Only edit:

```text
app/providers/amazon_provider.py
```

Map these SP-API operations to the provider dataclasses:

```text
get_product:
  GET /catalog/2022-04-01/items/{asin}
  GET /products/pricing/v0/price?Asins={asin}

get_competitor_snapshot:
  GET /products/pricing/v0/competitivePrice
  GET /products/pricing/v0/listings/{asin}/offers

update_price:
  PUT /listings/2021-08-01/items/{sellerId}/{sku}
  or Feeds API POST /feeds/2021-06-30/feeds for bulk updates

list_products:
  GET /catalog/2022-04-01/items?keywords=...
```

## 4. Factory Auto-Switches

No API route, schema, or repricing engine changes are needed. The factory chooses the provider:

```text
Amazon credentials present -> AmazonSPAPIProvider
Otherwise -> KaggleProvider
```

## 5. Test With dry_run=True First

Start with:

```json
{
  "asin": "YOUR_ASIN",
  "strategy": "auto",
  "dry_run": true
}
```

After validating recommendations and guardrails, use `dry_run=false` for real price updates.
