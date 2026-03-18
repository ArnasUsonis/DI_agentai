import feedparser
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

#model = ChatOllama(model="llama3.2", temperature=0.1).bind(tool_choice="required")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
).bind(tool_choice="required")

delfi_linkai = {
    "sportas":   "https://feed.delfi.lt/v2/channel/f59aa234-4a2b-11ed-94c9-0242c0a88103?format=rss",
    "krepšinis": "https://feed.delfi.lt/v2/channel/f52e2499-4a2b-11ed-94c9-0242c0a88103?format=rss",
}

@tool
def gauk_temas(tema: str) -> str:
    """Grąžina sporto temas iš naujienų portalo. tema: 'sportas' arba 'krepšinis'
    Args:
        tema: pagal užklausą nuspręsk kuri tema yra aktualiausia: 'sportas' ar 'krepšinis'.
    """
    url = delfi_linkai.get(tema.lower(), delfi_linkai["sportas"])
    feed = feedparser.parse(url)
    temos = []
    for entry in feed.entries:
        temos.append({
            "kategorija": entry.get("title", "nėra"),
            "url": entry.get("link", "nėra"),
        })
    return json.dumps(temos, ensure_ascii=False)


@tool
def gauk_straipsniu_pavadinimus(linkai: list[str]) -> str:
    """Gauna straipsnių pavadinimus iš nurodytų kategorijų URL sąrašo.
    Args:
        linkai: Sąrašas URL iš gauk_temas rezultatų. Pasirink 2 arba 3 aktualius URL pagal užklausą.
    """
    rezultatai = []
    
    for linkas in linkai:
        straipsniai = feedparser.parse(linkas)
        for entry in straipsniai.entries:
            rezultatai.append({
                "straipsnio pavadinimas": entry.get("title", "nėra"),
                "url": entry.get("link", "nėra"),
            })
    
    return json.dumps(rezultatai, ensure_ascii=False)


@tool
def gauk_straipsniu_informacija(straipsnio_linkai: list[str]) -> str:
    """Gauna pilną straipsnio informaciją iš naujienų portalo. Naudok kai reikia daugiau informacijos tam tikra tema.
    Args:
        straipsnio_linkas: Sąrašas URL iš gauk_straipsniu_pavadinimus rezultatų. Pasirink 2 arba 3 aktualius URL pagal užklausą.
    """
    straipsniai = []
    
    for linkas in straipsnio_linkai: 
        loader = WebBaseLoader(linkas)
        docs = loader.load()
        straipsniai.append({
            "straipsnio informacija": docs[0].page_content[:1000]
        })
    
    return json.dumps(straipsniai, ensure_ascii=False)


agent = create_agent(
    model=model,
    tools=[gauk_temas, gauk_straipsniu_pavadinimus, gauk_straipsniu_informacija],
    system_prompt="""
Tu esi sporto naujienų apžvalgininkas.

Tavo darbo eiga:
1. Pirmiausia naudok 'gauk_temas', kad gautum naujausias temas.
2. Atsirink temas, kurios geriausiai atitinka vartotojo užklausą. 
3. Naudok 'gauk_straipsniu_pavadinimus', kad gautum straipsnių pavadinimus aktualių temų.
4. Atsirink straipsnių pavadinimus kurie geriausiai atitinka užklausą.
5. Naudok 'gauk_straipsniu_informacija' gauti detalesnę informaciją tam tikro straipsnio.
6. Pateik apibendrintą, glaustą informaciją.

Visą reikalingą informaciją gauti naudok 'gauk_temas', 'gauk_straipsniu_pavadinimus' ir 'gauk_straipsniu_informacija' funkcijas.
"""
)

atsakymas = agent.invoke({
    "messages": [
        {
            "role": "user", 
            "content": "Kokios naujienos futbole?"
        }
    ]
})

print(atsakymas["messages"][-1].content)