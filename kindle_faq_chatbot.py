"""
KindleBot - FAQ Chatbot for Amazon Kindle
RAG pipeline using LangChain, ChromaDB, and Anthropic Claude API

Requirements:
    pip install langchain langchain-anthropic langchain-chroma chromadb
                langchain-community sentence-transformers anthropic

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python kindle_faq_chatbot.py
"""

import os
import sys

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

FAQ_DATA = [
    {
        "question": "How do I set up my new Kindle for the first time?",
        "answer": (
            "Charge it fully using the included USB cable. Power on with the bottom edge "
            "button. Select your language, connect to Wi-Fi, then sign in with your Amazon "
            "account. Your Kindle registers automatically and syncs previously purchased content."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "How do I register my Kindle to my Amazon account?",
        "answer": (
            "Go to Settings > Your Account > Register. Enter your Amazon email and password. "
            "If two-step verification is on, enter the OTP sent to your phone. Once registered, "
            "your Kindle appears in Manage Your Content and Devices at amazon.com."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "Can I use my Kindle without an Amazon account?",
        "answer": (
            "Basic reading works without an account if you sideload books via USB. "
            "However, purchasing from the Kindle Store, syncing across devices, and "
            "Whispersync all require an Amazon account."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "How do I deregister my Kindle?",
        "answer": (
            "Go to Settings > Your Account > Deregister Device and confirm. Alternatively, "
            "sign in to amazon.com > Manage Your Content and Devices > Devices, find your "
            "Kindle, and choose Deregister."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "What do I do if my Kindle will not connect to Wi-Fi during setup?",
        "answer": (
            "Kindle does not support enterprise (802.1X) networks. Restart your router, "
            "forget the network on Kindle (Settings > Wi-Fi > tap network > Forget), then "
            "reconnect. Try a mobile hotspot if the issue persists."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "How do I set up parental controls on Kindle?",
        "answer": (
            "Go to Settings > Parental Controls > Enable Parental Controls and create a PIN. "
            "You can restrict the Kindle Store, web browser, and cloud access. Amazon Kids "
            "provides a more comprehensive child-safe environment with curated content."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "Can I register multiple Kindles to the same Amazon account?",
        "answer": (
            "Yes. Amazon allows up to six Kindle devices and apps registered to one account. "
            "Books purchased once are available on all registered devices through the cloud."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "How do I rename my Kindle device?",
        "answer": (
            "On the device go to Settings > Your Account > Device Name. Type your preferred "
            "name and tap OK. You can also rename it at amazon.com under Manage Your Content "
            "and Devices > Devices."
        ),
        "category": "Setup & Registration",
    },
    {
        "question": "Can I set up multiple Amazon accounts on one Kindle?",
        "answer": (
            "A Kindle can only be registered to one Amazon account at a time. To switch, "
            "deregister first then re-register with the new account. Content from the "
            "previous account will no longer be accessible on the device."
        ),
        "category": "Setup & Registration",
    },

    {
        "question": "How long does Kindle battery last?",
        "answer": (
            "Kindle Paperwhite (11th gen) lasts up to 10 weeks with wireless off and "
            "30 minutes of daily reading at medium brightness. Kindle Scribe lasts up to "
            "12 weeks. Heavy use, Wi-Fi always on, or max brightness reduces these figures significantly."
        ),
        "category": "Battery & Charging",
    },
    {
        "question": "How do I charge my Kindle?",
        "answer": (
            "Connect the included USB-C cable (micro-USB on older models) to a 5W USB wall "
            "adapter or computer. A fully discharged Kindle takes about 4 hours to charge. "
            "A charging icon appears in the top-right corner while connected."
        ),
        "category": "Battery & Charging",
    },
    {
        "question": "Why is my Kindle not charging?",
        "answer": (
            "Try a different cable and adapter since cables are a common failure point. Clean "
            "the USB port gently. Use a wall adapter rather than a computer USB port. "
            "If the screen shows a critical battery icon, leave it plugged in for 30 minutes "
            "before trying to power on."
        ),
        "category": "Battery & Charging",
    },
    {
        "question": "How do I extend Kindle battery life?",
        "answer": (
            "Enable Airplane Mode when not purchasing or syncing. Reduce screen brightness. "
            "Set sleep mode to activate after 5 minutes (Settings > Device Options > Display). "
            "Disable automatic page refresh. Avoid leaving Wi-Fi on when reading locally stored books."
        ),
        "category": "Battery & Charging",
    },
    {
        "question": "Does Kindle support wireless charging?",
        "answer": (
            "Only the Kindle Paperwhite Signature Edition supports Qi wireless charging. "
            "All other Kindle models require a wired USB-C or micro-USB connection."
        ),
        "category": "Battery & Charging",
    },
    {
        "question": "Can I use a fast charger with my Kindle?",
        "answer": (
            "Kindle accepts standard 5W USB charging. A higher-wattage adapter will not "
            "damage the device but will not charge it faster than its rated speed. "
            "A standard 5W adapter is sufficient."
        ),
        "category": "Battery & Charging",
    },
    {
        "question": "Why does my Kindle battery drain fast?",
        "answer": (
            "Common causes: Wi-Fi always on, high brightness, background downloads, and "
            "frequent syncing. Check Settings > Account > Whispersync for Books. "
            "A factory reset followed by re-registration often resolves firmware-level drain issues."
        ),
        "category": "Battery & Charging",
    },

    {
        "question": "How do I buy a Kindle book?",
        "answer": (
            "On the device, open the Kindle Store from the home screen, browse or search, "
            "then tap the price. On amazon.com select the Kindle edition and choose your "
            "device from 'Send to device'. The book downloads over Wi-Fi automatically."
        ),
        "category": "Content & Books",
    },
    {
        "question": "How do I borrow Kindle books from the library?",
        "answer": (
            "Use the OverDrive or Libby app, or visit your library's digital collection "
            "online. Borrow the title and choose 'Send to Kindle'. You need an Amazon "
            "account linked to your library card. Borrowed titles expire automatically."
        ),
        "category": "Content & Books",
    },
    {
        "question": "Can I read PDF files on my Kindle?",
        "answer": (
            "Yes. Transfer PDFs via USB by copying to the Documents folder, or email them "
            "to your Kindle email address found in Settings > Your Account > Send-to-Kindle "
            "Email. PDFs display as-is; use landscape mode for better readability."
        ),
        "category": "Content & Books",
    },
    {
        "question": "What file formats does Kindle support?",
        "answer": (
            "Kindle natively supports AZW3, MOBI, PDF, TXT, and HTML. It also supports "
            "DOCX, JPEG, GIF, PNG, and BMP via the Send-to-Kindle service, which converts "
            "files automatically. EPUB files require conversion using a tool like Calibre."
        ),
        "category": "Content & Books",
    },
    {
        "question": "How do I transfer books from my computer to Kindle via USB?",
        "answer": (
            "Connect Kindle to your computer with a USB cable. It appears as a removable "
            "drive. Open it and navigate to the 'documents' folder. Copy your compatible "
            "ebook files there. Safely eject the Kindle and books appear on the home screen."
        ),
        "category": "Content & Books",
    },
    {
        "question": "How do I delete a book from my Kindle?",
        "answer": (
            "Press and hold the book cover on the home screen, then select 'Remove from "
            "Device'. This removes the local copy but keeps it in your cloud library. "
            "To permanently delete a purchase, go to amazon.com > Manage Your Content "
            "and Devices and delete from there."
        ),
        "category": "Content & Books",
    },
    {
        "question": "What is Kindle Unlimited?",
        "answer": (
            "Kindle Unlimited is a subscription service giving access to over 4 million "
            "titles, audiobooks, and magazines. You can have up to 20 titles borrowed at "
            "once. It is separate from Amazon Prime Reading, which offers a smaller "
            "curated selection included with Prime."
        ),
        "category": "Content & Books",
    },
    {
        "question": "How do I send documents to my Kindle using email?",
        "answer": (
            "Find your Kindle email address at Settings > Your Account > Send-to-Kindle "
            "Email. Add your personal email as an approved sender at amazon.com > Manage "
            "Your Content and Devices > Preferences > Personal Document Settings. "
            "Then email the file as an attachment to your Kindle address."
        ),
        "category": "Content & Books",
    },
    {
        "question": "Can I read Kindle books on my phone or tablet?",
        "answer": (
            "Yes. Download the free Kindle app for iOS or Android. Sign in with your "
            "Amazon account. All purchased books sync across your Kindle device and the app "
            "via Whispersync, including your last page read and highlights."
        ),
        "category": "Content & Books",
    },

    {
        "question": "How do I cancel Kindle Unlimited?",
        "answer": (
            "Go to amazon.com > Account and Lists > Memberships and Subscriptions > "
            "Kindle Unlimited > Manage Membership > Cancel Kindle Unlimited. "
            "Your access continues until the end of the current billing period."
        ),
        "category": "Account & Subscriptions",
    },
    {
        "question": "How do I update my payment method for Kindle purchases?",
        "answer": (
            "Go to amazon.com > Account and Lists > Your Account > Manage Payment Methods. "
            "Add or edit a card. Your default 1-Click payment method is used for Kindle "
            "Store purchases unless you select another at checkout."
        ),
        "category": "Account & Subscriptions",
    },
    {
        "question": "How do I share Kindle books with family?",
        "answer": (
            "Use Amazon Household. Go to amazon.com > Account > Amazon Household and invite "
            "a family member. Adults in the household can share eligible purchased books. "
            "Not all titles are eligible since publisher restrictions may apply."
        ),
        "category": "Account & Subscriptions",
    },
    {
        "question": "What is Amazon Household and how does it work with Kindle?",
        "answer": (
            "Amazon Household lets up to two adults, four teens, and four children share "
            "Prime benefits and eligible digital content. Kindle books shared through "
            "Household appear in each member's library. Each person still has a separate "
            "Amazon account and reading progress."
        ),
        "category": "Account & Subscriptions",
    },
    {
        "question": "How do I get a refund for a Kindle book?",
        "answer": (
            "Go to amazon.com > Returns and Orders, find the Kindle purchase, and select "
            "'Return for Refund'. Amazon typically allows returns within 7 days of purchase "
            "for books that have not been substantially read. The book is removed from "
            "your library upon refund."
        ),
        "category": "Account & Subscriptions",
    },
    {
        "question": "How do I view my Kindle purchase history?",
        "answer": (
            "Go to amazon.com > Manage Your Content and Devices > Content. Filter by "
            "Books, Newspapers, or other types. You can also see order history under "
            "Account and Lists > Returns and Orders and filter by digital orders."
        ),
        "category": "Account & Subscriptions",
    },
    {
        "question": "Does Amazon Prime include free Kindle books?",
        "answer": (
            "Amazon Prime includes Prime Reading, which gives access to a rotating "
            "selection of over 3,000 books, magazines, and comics at no extra cost. "
            "This is separate from Kindle Unlimited, which has a much larger catalog "
            "but costs extra beyond Prime."
        ),
        "category": "Account & Subscriptions",
    },

    {
        "question": "How do I change the font size on my Kindle?",
        "answer": (
            "While reading, tap the center of the screen to bring up the toolbar, "
            "then tap the font icon (Aa). A menu lets you adjust font size, typeface, "
            "line spacing, and margins. Changes apply immediately to the current book."
        ),
        "category": "Reading Features",
    },
    {
        "question": "How do I use the Kindle dictionary?",
        "answer": (
            "Press and hold a word while reading. A pop-up shows the definition from "
            "the built-in dictionary. Tap 'Full Definition' for more detail. You can "
            "change the default dictionary under Settings > Language and Dictionaries > Dictionaries."
        ),
        "category": "Reading Features",
    },
    {
        "question": "How do I highlight text and add notes on Kindle?",
        "answer": (
            "Press and hold the start of a passage, then drag to the end. Release to "
            "see options: Highlight, Note, Share, or Search. Choose a highlight color "
            "if desired. All highlights and notes sync to your account and are viewable "
            "at read.amazon.com/notebook."
        ),
        "category": "Reading Features",
    },
    {
        "question": "What is Whispersync and how does it work?",
        "answer": (
            "Whispersync automatically syncs your last page read, bookmarks, highlights, "
            "and notes across all devices and apps registered to your Amazon account. "
            "It works over Wi-Fi in the background. Enable or disable it at "
            "Settings > Account > Whispersync for Books."
        ),
        "category": "Reading Features",
    },
    {
        "question": "How do I enable dark mode on Kindle?",
        "answer": (
            "Swipe down from the top of the screen to open Quick Settings, then tap "
            "Dark Mode. Alternatively, go to Settings > Display > Dark Mode. "
            "Dark mode inverts the display to white text on a black background, "
            "which many readers find easier on the eyes at night."
        ),
        "category": "Reading Features",
    },
    {
        "question": "How do I use audiobooks on Kindle?",
        "answer": (
            "Kindle devices do not have built-in text-to-speech for ebooks. "
            "For audiobooks, use the Audible app or purchase Audible content on "
            "a Kindle Paperwhite or Scribe, which have Bluetooth audio output. "
            "Pair Bluetooth headphones via Settings > Bluetooth."
        ),
        "category": "Reading Features",
    },
    {
        "question": "How do I search inside a book on Kindle?",
        "answer": (
            "While reading, tap the top of the screen to show the toolbar, then tap "
            "the search (magnifying glass) icon. Type your search term. Results show "
            "matches within the book along with page context. You can also search "
            "Wikipedia or the Kindle Store from the same menu."
        ),
        "category": "Reading Features",
    },
    {
        "question": "Can I read Kindle books in landscape mode?",
        "answer": (
            "Yes. While reading, tap the center of the screen, tap the font icon (Aa), "
            "then select Page Display > Landscape. This is especially useful for PDFs "
            "and technical documents with wide layouts."
        ),
        "category": "Reading Features",
    },
    {
        "question": "How do I create a collection or organize books on Kindle?",
        "answer": (
            "From the home screen, press and hold a book cover, then select 'Add to List'. "
            "Create a new list or add to an existing one. Lists appear in your library "
            "view and sync across devices."
        ),
        "category": "Reading Features",
    },

    {
        "question": "How do I restart my Kindle?",
        "answer": (
            "Press and hold the power button for 9 seconds until the screen goes blank, "
            "then release. The Kindle restarts automatically. This resolves most freezing "
            "and performance issues without affecting your content or settings."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "How do I factory reset my Kindle?",
        "answer": (
            "Go to Settings > Device Options > Reset Device and confirm. This erases all "
            "content and settings, returning the device to factory state. Your purchased "
            "content remains in the cloud and re-downloads after you sign back in. "
            "Use this as a last resort for persistent issues."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "My Kindle screen is frozen. What do I do?",
        "answer": (
            "Hold the power button for 40 seconds to force a hard restart. If the "
            "screen remains frozen after restarting, connect to power and try again. "
            "If a specific book causes freezing, delete and re-download it since the "
            "file may be corrupted."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "Why are my Kindle books not downloading?",
        "answer": (
            "Check your Wi-Fi connection at Settings > Wi-Fi. Make sure Airplane Mode is off. "
            "Confirm the book is assigned to your device at amazon.com > Manage Your Content "
            "and Devices. Try restarting the Kindle. If the issue continues, deregister and "
            "re-register the device."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "How do I update Kindle software?",
        "answer": (
            "Connect to Wi-Fi and go to Settings > Device Options > Update Your Kindle. "
            "If grayed out, the device is already on the latest version. Updates also "
            "install automatically overnight when connected to Wi-Fi and charging."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "What do I do if my Kindle touchscreen is not responding?",
        "answer": (
            "Restart the Kindle by holding the power button for 9 seconds. Remove any "
            "screen protector, which can interfere with touch sensitivity. Clean the screen "
            "with a dry microfiber cloth. If the issue persists after a restart, perform "
            "a factory reset."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "Why does my Kindle show a black screen?",
        "answer": (
            "A completely black screen usually means the battery is fully depleted. "
            "Connect to power and wait 30 minutes before pressing the power button. "
            "If the screen remains black after charging, hold the power button for "
            "40 seconds to force restart."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "My Kindle is not recognized by my computer. How do I fix this?",
        "answer": (
            "Try a different USB cable since many cables are charge-only and do not transfer data. "
            "Use a direct USB port rather than a hub. On Windows, check Device Manager for "
            "driver errors. On Mac, the Kindle appears as a drive in Finder. "
            "Restarting both the Kindle and computer often resolves the issue."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "How do I contact Amazon Kindle customer support?",
        "answer": (
            "Go to amazon.com > Help > Contact Us. Choose Kindle as the topic. "
            "Options include live chat, phone callback, and email. You can also visit "
            "the Kindle community forums at amazon.com/forum/kindle for peer support. "
            "Amazon support is available 24 hours a day, 7 days a week."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "Why does my Kindle say Device storage is full?",
        "answer": (
            "Remove books you have finished reading by pressing and holding the cover "
            "and selecting Remove from Device. They remain in the cloud. "
            "Kindle Paperwhite has 8GB or 16GB of storage. "
            "Avoid sideloading large PDF or image-heavy files if storage is limited."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "How do I fix Kindle Wi-Fi that keeps disconnecting?",
        "answer": (
            "Go to Settings > Wi-Fi, forget the network, then reconnect. "
            "Check that your router firmware is up to date. Try switching between "
            "2.4 GHz and 5 GHz bands. If Kindle disconnects only during sleep, "
            "go to Settings > Wireless and ensure Wi-Fi is set to stay on."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "How do I fix a Kindle that will not turn on?",
        "answer": (
            "Connect Kindle to power for at least 30 minutes first. Then hold the power "
            "button for 40 seconds. If you see a battery icon, it needs more charge time. "
            "If there is no response after an hour of charging, contact Amazon support "
            "as the battery may need replacement."
        ),
        "category": "Troubleshooting",
    },
    {
        "question": "What do I do if my Kindle is stuck on the startup screen?",
        "answer": (
            "Hold the power button for 40 seconds to force restart. If it repeatedly "
            "gets stuck on startup, perform a factory reset by holding the power button "
            "while the device is connected to a computer via USB, which forces recovery mode."
        ),
        "category": "Troubleshooting",
    },
]

def build_documents(faq_data: list) -> list:
    """Convert raw FAQ dicts into LangChain Documents."""
    documents = []
    for i, item in enumerate(faq_data):
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        doc = Document(
            page_content=text,
            metadata={
                "source": f"faq_{i}",
                "category": item["category"],
                "question": item["question"],
            },
        )
        documents.append(doc)
    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into chunks for embedding.
    chunk_size is generous to avoid splitting a single Q&A across chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list) -> Chroma:
    """
    Embed document chunks with a local HuggingFace model and store in ChromaDB.
    Using a local embedding model avoids a second paid API dependency.
    The model (~90 MB) downloads automatically on first run.
    """
    print("Loading embedding model (downloads ~90 MB on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"Embedding {len(chunks)} document chunks into ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="kindle_faq",
        persist_directory="./kindle_chroma_db",
    )
    print("Vector store ready.\n")
    return vector_store


SYSTEM_PROMPT = (
    "You are KindleBot, a friendly and knowledgeable support assistant "
    "for Amazon Kindle devices and services.\n\n"
    "Answer the user's question using ONLY the context provided below. "
    "The context contains relevant excerpts from the official Kindle FAQ.\n\n"
    "Rules:\n"
    "- If the context contains enough information, give a clear and helpful answer.\n"
    "- If the context does not contain enough information, say: "
    "'I don't have specific information about that. Please visit amazon.com/help "
    "or contact Amazon customer support.'\n"
    "- Do not make up information that is not in the context.\n"
    "- Keep answers concise and practical.\n\n"
    "Context:\n{context}"
)


def build_rag_chain(vector_store: Chroma):
    """Assemble the retrieval-augmented generation chain using LangChain LCEL."""

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.2,
        max_tokens=512,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "User question: {question}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


WELCOME_BANNER = """
+------------------------------------------------------+
|         KindleBot -- Kindle FAQ Assistant            |
|  Ask me anything about your Kindle device or account |
|           Type 'quit' or 'exit' to leave             |
+------------------------------------------------------+

Example questions:
  - How do I charge my Kindle?
  - Why is my Kindle screen frozen?
  - How do I borrow library books?
  - Can I share books with family?
  - How do I change the font size?
"""


def run_chatbot(chain, retriever) -> None:
    print(WELCOME_BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("KindleBot: Thanks for using KindleBot. Happy reading!")
            break

        # Retrieve source documents for transparency
        source_docs = retriever.invoke(user_input)
        categories = list({doc.metadata.get("category", "General") for doc in source_docs})

        print("\nKindleBot: ", end="", flush=True)
        answer = chain.invoke(user_input)
        print(answer)
        print(f"\n  [Sources consulted: {', '.join(categories)}]\n")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with:  export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print("Building KindleBot knowledge base...")

    documents = build_documents(FAQ_DATA)
    chunks    = split_documents(documents)
    store     = build_vector_store(chunks)
    chain, retriever = build_rag_chain(store)

    run_chatbot(chain, retriever)


if __name__ == "__main__":
    main()
