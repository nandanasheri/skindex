import requests
from bs4 import BeautifulSoup

'''
Scrape Plan
Scrape 400 - 500 products
each product
{
"docid":
"title":
"brand":
"desc":
"key_ingredient_func" : [list of IDs]
"other_ingredient_func" : []
"image_url" : 
}

each ingredient function
{
"func_id":
"title":
"desc":
}
Scraping about 60-70 pages will get us approximately 400 unique products
'''
def scrape_products(pages:int):
    exclude_links = set(["/products/new", "/products/create"])
    all_product_links = set()

    for i in range(1, pages+1):
        url = f"https://incidecoder.com/products?page={i}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        product_links = soup.find_all('a')        
        for a in product_links:
            href = a.get("href")  # or a['href']
            if href.startswith("/products/") and href not in exclude_links and "discontinued" not in href:
                all_product_links.add(href)
    
    return all_product_links

def scrape_each_product(product_links:list):
    for link in product_links:
        url = f"https://incidecoder.com{link}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        print(soup)

# products = scrape_products(30)
scrape_each_product(["/products/ultrasun-anti-age-spf-50"])
