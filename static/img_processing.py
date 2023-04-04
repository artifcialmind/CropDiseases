import bs4
from bs4 import BeautifulSoup
def process():
    html = "../static/display.html"
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find(id="fileUpload")
    return element