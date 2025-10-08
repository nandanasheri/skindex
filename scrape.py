import requests
from bs4 import BeautifulSoup

url = "https://incidecoder.com/products/the-ordinary-niacinamide-10-zinc-1"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")
print(soup)