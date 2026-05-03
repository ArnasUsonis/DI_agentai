import feedparser
import json
import requests
from bs4 import BeautifulSoup
from datetime import date
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
import logging
import os
import warnings
import sys
import io
from langchain_core.globals import set_debug
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import trim_messages

SCOPES = ["https://www.googleapis.com/auth/calendar"]

set_debug(True) # ziureti galvojimo procesa agento

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  
os.environ["HF_HUB_VERBOSITY"] = "error"

stderr = sys.stderr
sys.stderr = io.StringIO()

warnings.filterwarnings("ignore")                 

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR) # cia tiesiog paslepiame nereikalinga info kaip loading weights ar erorus

class Rungtynes(BaseModel):
    komanda1: str = Field(description="First team name")
    komanda2: str = Field(description="Second team name")
    data: str = Field(description="Match date in YYYY-MM-DD format")
    laikas: str | None = Field(default=None, description="Match time in HH:MM format, if available")

class RungtyniuSarasas(BaseModel):
    rungtynes: List[Rungtynes]


LLM_NAME = "qwen2.5" # modelio pavadinimas kuri naudosime

model = ChatOllama(model=LLM_NAME, temperature=0).bind(tool_choice="required")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # semantines paieskos modelis
sys.stderr = stderr


rss_sources = {
    "football":         "https://feed.delfi.lt/v2/articles/88690?format=rss",
    "nba":              "https://feed.delfi.lt/v2/articles/72876174?format=rss",
    "euroleague":       "https://feed.delfi.lt/v2/articles/93050597?format=rss",
    "lkl":              "https://feed.delfi.lt/v2/articles/93050651?format=rss",
    "basketball_champions_league":   "https://feed.delfi.lt/v2/articles/93050691?format=rss",
}

def leidimas_naudotis_kaledoriumi(): # suteikiam leidima kodui naudotis google kalendoriumi, pirma karta paleidus issoks langas kuriame reikes prisijungti
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


@tool
def gauk_straipsniu_sarasa(kategorija: str, uzklausa: str) -> str:
    """ALWAYS call this tool FIRST. Finds relevant sports news articles by user query.
    Args:
        kategorija: sports category – one of: 'football', 'nba', 'euroleague', 'lkl', 'basketball_champions_league'.
        uzklausa: user query.
    """
    url = rss_sources.get(kategorija.lower())
    if not url: #leidziam modeliui pasitaisyti jei parenka ne ta kategorija
        return f"Unknown category: '{kategorija}'. Available categories: {list(rss_sources.keys())}"

    feed = feedparser.parse(url)
    docs = []
    for entry in feed.entries:
        docs.append(Document(
            page_content=f"{entry.get('title')}. {entry.get('description')}",
            metadata={"url": entry.get("link")}
        ))

    db = FAISS.from_documents(docs, embeddings) # is straipsniu sukuria sukuria vektoriu duomenu baze

    rezultatai = db.similarity_search(uzklausa, k=5)
    #aktualus_rezultatai = [r for r, score in rezultatai if score < 1.5] # tikrinam ar straipsniai yra tikrai pakankamai aktualus pries ivairove
    #if not aktualus_rezultatai:
        #return "No relevant articles found for this query."

    #rezultatai = db.max_marginal_relevance_search(uzklausa, k=3, fetch_k=15) # suranda aktualius straipsnius bet su ivairove

    atrinkti = [ # aktualus straipsniai
        {"pavadinimas": r.page_content, "url": r.metadata["url"]}
        for r in rezultatai
    ]
    return json.dumps(atrinkti, ensure_ascii=False)


@tool
def gauk_straipsnio_teksta(linkai: list[str], uzklausa: str) -> str:
    """Fetches full article text and returns the most relevant chunks by user query.
    Call this tool AFTER gauk_straipsniu_sarasa.
    ALWAYS pass ALL URLs received from gauk_straipsniu_sarasa, not just one.
    Args:
        linkai: list of ALL article URLs from gauk_straipsniu_sarasa results.
        uzklausa: user query.
    """
    docs = []
    for url in linkai:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            tekstas = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))
            tekstas = tekstas.replace("\xa0", " ")
            if tekstas:
                docs.append(Document(
                    page_content=tekstas,
                    metadata={"url": url}
                ))
        except Exception as e:
            print(f"Klaida: {url}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # chunksais dalinam visa teksta 
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    rezultatai = db.similarity_search(uzklausa, k=3)

    aktualus = [ # aktuali informacija is tu pilnu straipsniu
        {"tekstas": r.page_content}
        for r in rezultatai
    ]
    return json.dumps(aktualus, ensure_ascii=False)


@tool
def isskirk_rungtynes(straipsniai_json: str) -> str:
    """Extracts upcoming match information (teams, date, time) from article text.
    Call this tool ALWAYS when user asks about upcoming matches or schedule.
    NEVER skip this tool if user asks about upcoming matches.
    Call this tool AFTER gauk_straipsnio_teksta.
    Args:
        straipsniai_json: article text from gauk_straipsnio_teksta results.
    """
    if isinstance(straipsniai_json, list): # buvo kad sarasu paduodavo tai konvertuojam i stringa
        straipsniai_json = json.dumps(straipsniai_json, ensure_ascii=False)

    llm = ChatOllama(model=LLM_NAME, temperature=0).with_structured_output(RungtyniuSarasas)
    rezultatas = llm.invoke(f"""
        Extract ONLY upcoming matches found in the text below.
        If no match information is found in the text - return empty list.
        Do NOT use any prior knowledge or training data.
        Only use information explicitly mentioned in the provided text.
        Convert dates to YYYY-MM-DD format.
        Convert time to HH:MM format.
        If date or time is not mentioned - leave empty.
        
        Text: {straipsniai_json}
    """)
    return rezultatas.model_dump_json()

@tool
def ideti_i_kalendoriu(rungtynes_json: str) -> str:
    """Adds upcoming matches to Google Calendar.
    Call this tool AFTER isskirk_rungtynes.
    Args:
        rungtynes_json: JSON string with match information from isskirk_rungtynes results.
    """
    try:
        service = leidimas_naudotis_kaledoriumi()
        rungtynes = json.loads(rungtynes_json)["rungtynes"]
        
        prideta = []
        for r in rungtynes:
            if r["laikas"]:
                pradzia = f"{r['data']}T{r['laikas']}:00"
                event = {
                    "summary": f"{r['komanda1']} vs {r['komanda2']}",
                    "start": {"dateTime": pradzia, "timeZone": "Europe/Vilnius"},
                    "end":   {"dateTime": pradzia, "timeZone": "Europe/Vilnius"},
                }
            else:
                event = {
                    "summary": f"{r['komanda1']} vs {r['komanda2']}",
                    "start": {"date": r["data"]},
                    "end":   {"date": r["data"]},
                }
            
            service.events().insert(calendarId="primary", body=event).execute()
            prideta.append(f"{r['komanda1']} vs {r['komanda2']} - {r['data']}")
        
        return f"Įdėta į kalendorių: {', '.join(prideta)}"
    
    except Exception as e:
        return f"Klaida įdedant į kalendorių: {str(e)}"


siandiena = date.today().isoformat()

memory = MemorySaver()
agent = create_react_agent(
    model=model,
    tools=[gauk_straipsniu_sarasa, gauk_straipsnio_teksta, isskirk_rungtynes, ideti_i_kalendoriu],
    prompt=f"""
You are a Lithuanian sports news assistant. Today's date: {siandiena}.

Always follow this EXACT order, never skip steps:
1. Call gauk_straipsniu_sarasa to find relevant articles.
2. ALWAYS call gauk_straipsnio_teksta with the URLs from step 1 before doing anything else.
3. If user asks about upcoming matches - call isskirk_rungtynes with the text from step 2.
4. If user asks to add matches to calendar - call ideti_i_kalendoriu AFTER isskirk_rungtynes.
5. Provide a summary in Lithuanian as a single continuous paragraph without any line breaks, bullet points or numbered lists.
""",
    checkpointer=memory, # trumpalaikes atminties implementacija
    messages_modifier=trim_messages(
        max_tokens=4,        # saugo 4 paskutinius pranesimus
        strategy="last",     # saugomi paskutiniai o ne pirmi pranesimai
        token_counter=len,   # kad tokenai skaiciuojami butu pranesimu kiekiu
        include_system=True, # atsimenami promptai
        allow_partial=False, 
        start_on="human",    # zmogaus uzklausas pirmas rodo
    )
)

#print(agent.get_graph().draw_mermaid()) # vizualizavimui

while True:
        uzklausa = input("Parašykite užklausą arba exit, kad baigti: ").strip()
            
        if uzklausa.lower() == "exit":
            break

        atsakymas = agent.invoke(
            {"messages": [{"role": "user", "content": uzklausa}]},
            config={"configurable": {"thread_id": "1"}}
        )

        tekstas = atsakymas["messages"][-1].content
        tekstas = " ".join(tekstas.split())
        print(tekstas)