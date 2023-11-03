import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://dataplatform.cloud.ibm.com/docs/content/wsj/wscommon/glossary-cpdaas.html?context=cpdaas&audience=wdp"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

h3_elements = soup.find_all("h3")
terms = [element.text.strip() for element in h3_elements]
df = pd.DataFrame({"terms": terms})
df.to_csv("ibm_glossary_terms.csv")
