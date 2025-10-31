import requests
from bs4 import BeautifulSoup
import re
import json

'''
Scrape Plan
Scrape 400 - 500 products
each product
{
"docid":
"title":
"brand":
"desc":
"product_url":
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

    for i in range(1, pages+1, 2):
        url = f"https://incidecoder.com/products?page={i}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        product_links = soup.find_all('a')    
        print("LEN of products", i, len(product_links))   
        for a in product_links:
            href = a.get("href")  # or a['href']
            if href.startswith("/products/") and href not in exclude_links and "discontinued" not in href:
                all_product_links.add(href)
            
    return all_product_links

def scrape_each_product(product_links:list):
    docid = 1
    all_products_json = []
    ingredients_json = []
    unique_ingredients = {}

    for link in product_links:
        prod_json = {}
        url = f"https://incidecoder.com{link}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

        prod_json['docid'] = docid
        prod_json['product_url'] = url

        ids = ["product-title", "product-brand-title", "product-details", "product-main-image"]
        classes = ["title", "brand", "desc", "img_url"]

        for i in range(len(ids)):
            each_id = ids[i]
            each_class = classes[i]
            soup_text = soup.find(id=each_id)

            if soup_text:
                text = soup_text.get_text()
                # specifically for scraping the image
                if each_id == "product-main-image":
                    product_img = soup_text.find("img")
                    img_url = product_img["src"] if product_img else None
                    prod_json[each_class] = img_url
                # specifically for scraping description since it has some garbage strings in the middle
                elif each_id == "product-details":
                    cleaned = " ".join(text.split())
                    # 2. Remove [more] and [less] or any [word] tokens
                    cleaned = re.sub(r"\[.*?\]", "", cleaned)
                    # 3. Clean extra spaces again
                    cleaned_desc = re.sub(r"\s+", " ", cleaned).strip()
                    prod_json[each_class] = cleaned_desc

                else:
                    prod_json[each_class] = text.strip()
        
        # for ingredient functions
        for block in soup.find_all("div", class_="ingredlist-by-function-block"):
            # Either returns Key Ingredients or Other Ingredients
            section_title = block.find("h3")
            if not section_title:
                continue
            section_title = section_title.get_text(strip=True)

            links = block.find_all("a", class_="func-link")

            func_ids_list = []
            for a in links:
                if a.get("href"):
                    function = a.get_text(strip=True)
                    func_url = a["href"]
                    # found a new ingredient - update json
                    if function not in unique_ingredients:                        
                        r = requests.get(f"https://incidecoder.com{func_url}")
                        func_soup = BeautifulSoup(r.text, "html.parser")
                        function_desc = func_soup.find("p").get_text(strip=True)
                        ingr_json = {"func_id" : len(ingredients_json), "title":function, "desc" : function_desc}
                        unique_ingredients[function] = len(ingredients_json)
                        ingredients_json.append(ingr_json)

                    func_ids_list.append(unique_ingredients[function])
            

            if "key" in section_title.lower():
                prod_json['key_ingredient_func'] = func_ids_list
            if "other" in section_title.lower():
                prod_json['other_ingredient_func'] = func_ids_list
        
        docid += 1
        all_products_json.append(prod_json)
    
    return all_products_json, ingredients_json

def write_json(products, ingredients):
    with open("inci_products.jsonl", "w", encoding="utf-8") as f:
        for product in products:
            json.dump(product, f, ensure_ascii=False)
            f.write("\n")
    
    with open("inci_ingredient_functions.jsonl", "w", encoding="utf-8") as f:
        for ingredient in ingredients:
            json.dump(ingredient, f, ensure_ascii=False)
            f.write("\n")
    
print("Scraping Product Links")
products = scrape_products(1000)
print("Statistics of Products Scraped")
print(f"Total Number of Products = {len(products)}")

print("Building JSON products and ingredient functions")
all_products, all_ingredients = scrape_each_product(products)

print("Statistics of Product JSON and Ingredients JSON")
print(f"Total Number of Products = {len(all_products)} (must be the same as number of products scraped)")
print(f"Total Number of Ingredient Functions = {len(all_ingredients)}")

print("Writing to JSON file")
write_json(all_products, all_ingredients)
