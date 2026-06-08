# Amazon Bulk Upload Demo

Files:

- `amazon_seller_bulk_upload_demo.csv`

What it is:

- A generic demo CSV for Amazon Seller bulk listing uploads
- Useful for testing mapping from `product-ai-agent` output into a flat-file style export

What it is not:

- It is not an official Amazon category-specific flat file
- Amazon often requires marketplace-specific and category-specific templates from Seller Central
- Some categories need extra attributes beyond this demo

Field notes:

- `product-id-type=4` means UPC in this demo
- `update-delete=Update` is used for create/update style imports
- `feed-product-type` is a simplified generic value for demonstration
- image URLs must be publicly reachable by Amazon when used in real uploads

Suggested next step:

- If you want, I can build an export endpoint in `product-ai-agent` that converts generated product drafts into this CSV format automatically.
