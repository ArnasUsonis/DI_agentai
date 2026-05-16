import feedparser
import requests
from bs4 import BeautifulSoup
import os
import json
import re

rss_sources = {
    "football":                   "https://feed.delfi.lt/v2/articles/88690?format=rss",
    "nba":                        "https://feed.delfi.lt/v2/articles/72876174?format=rss",
    "euroleague":                 "https://feed.delfi.lt/v2/articles/93050597?format=rss",
    "lkl":                        "https://feed.delfi.lt/v2/articles/93050651?format=rss",
    "basketball_champions_league": "https://feed.delfi.lt/v2/articles/93050691?format=rss",
}

for kategorija, url in rss_sources.items():
    feed = feedparser.parse(url)
    issaugota = 0
    
    for i, entry in enumerate(feed.entries):
        try:
            r = requests.get(entry.link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            tekstas = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
            tekstas = re.sub(r'\s+', ' ', tekstas).strip()  # pašaliname visus whitespace simbolius
            
            failo_kelias = f"{kategorija}/straipsnis_{i+1}.json"
            
            with open(failo_kelias, "w", encoding="utf-8") as f:
                json.dump({
                    "pavadinimas": entry.title,
                    "aprasymas": entry.get("description", ""),
                    "straipsnis": tekstas
                }, f, ensure_ascii=False, indent=2)
            
            issaugota += 1
        except Exception as e:
            print(f"Klaida {kategorija}/straipsnis_{i+1}: {e}")
    
    print(f"{kategorija}: {issaugota} straipsniai išsaugoti")