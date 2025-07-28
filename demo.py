import re
import os
import json
from typing import List, Dict, Any
from pathlib import Path
from google.colab import userdata
from langchain_openai import OpenAIEmbeddings #Embeddings - Doc: https://python.langchain.com/docs/integrations/text_embedding/openai/
from langchain_chroma import Chroma # VectorDB - Doc: https://python.langchain.com/docs/integrations/vectorstores/chroma/#basic-initialization
from langchain_openai import ChatOpenAI #LLM - Doc: https://python.langchain.com/docs/integrations/chat/openai/
from langchain_community.document_loaders import PyPDFLoader #PDF loader - Doc: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Splitter / Chunks - Doc: https://python.langchain.com/docs/how_to/recursive_text_splitter/
import gradio as gr #Gradio - Doc: https://www.gradio.app/main/docs/gradio/api
from pickle import TRUE
import numpy as np

drivePath = "Project/"
sourceFiles = drivePath + "SourceFiles"
sourceJson = drivePath + "SourceJson"
vectorDBPath = drivePath + "Vectordb"

os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

#Doc: https://python.langchain.com/docs/integrations/text_embedding/
embedding_model = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

####INITIALIZATION#####
# Create the source files directory
if not os.path.exists(drivePath):
  os.mkdir(drivePath)
if not os.path.exists(sourceFiles):
  os.mkdir(sourceFiles)
if not os.path.exists(sourceJson):
  os.mkdir(sourceJson)

#Doc: https://python.langchain.com/docs/integrations/vectorstores/
#Here the Vector DB chosen was Chroma
#VDB for MTG Rulling
vector_db_MTG = Chroma(
    collection_name="MTG-embedding-3-small_v1",
    embedding_function=embedding_model,
    persist_directory=vectorDBPath,
    create_collection_if_not_exists = TRUE
)

vector_db_Cards = Chroma(
    collection_name="MTGCards-embedding-3-small_v2",
    embedding_function=embedding_model,
    persist_directory=vectorDBPath,
    create_collection_if_not_exists = TRUE
)

#Doc: https://python.langchain.com/docs/integrations/chat/openai/
#Here the model chosen was gpt-4o
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.6
)

llm_query_transformation = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.8
)

prompt_system = f"""
  Instructions:
  Answer the provided user query by using the provided context, chat history, or both to deduce the answer.
  The chat history is ordered from the first to the latest interaction, the user is a message from the user, the assistant is a message from the LLM.
  If you don't know the answer, say 'I don't know, sorry.'
"""
messages = [
    ("system", prompt_system)
]

# List of acceptable sections and subsections
SECTIONS = [
    "1. Game Concepts",
    "2. Parts of a Card",
    "3. Card Types",
    "4. Zones",
    "5. Turn Structure",
    "6. Spells, Abilities, and Effects",
    "7. Additional Rules",
    "8. Multiplayer Rules",
    "9. Casual Variants"
]

SUBSECTIONS = [
    # Section 1
    "100. General",
    "101. The Magic Golden Rules",
    "102. Players",
    "103. Starting the Game",
    "104. Ending the Game",
    "105. Colors",
    "106. Mana",
    "107. Numbers and Symbols",
    "108. Cards",
    "109. Objects",
    "110. Permanents",
    "111. Tokens",
    "112. Spells",
    "113. Abilities",
    "114. Emblems",
    "115. Targets",
    "116. Special Actions",
    "117. Timing and Priority",
    "118. Costs",
    "119. Life",
    "120. Damage",
    "121. Drawing a Card",
    "122. Counters",
    "123. Stickers",
    # Section 2
    "200. General",
    "201. Name",
    "202. Mana Cost and Color",
    "203. Illustration",
    "204. Color Indicator",
    "205. Type Line",
    "206. Expansion Symbol",
    "207. Text Box",
    "208. Power/Toughness",
    "209. Loyalty",
    "210. Defense",
    "211. Hand Modifier",
    "212. Life Modifier",
    "213. Information Below the Text Box",
    # Section 3
    "300. General",
    "301. Artifacts",
    "302. Creatures",
    "303. Enchantments",
    "304. Instants",
    "305. Lands",
    "306. Planeswalkers",
    "307. Sorceries",
    "308. Kindreds",
    "309. Dungeons",
    "310. Battles",
    "311. Planes",
    "312. Phenomena",
    "313. Vanguards",
    "314. Schemes",
    "315. Conspiracies",
    # Section 4
    "400. General",
    "401. Library",
    "402. Hand",
    "403. Battlefield",
    "404. Graveyard",
    "405. Stack",
    "406. Exile",
    "407. Ante",
    "408. Command",
    # Section 5
    "500. General",
    "501. Beginning Phase",
    "502. Untap Step",
    "503. Upkeep Step",
    "504. Draw Step",
    "505. Main Phase",
    "506. Combat Phase",
    "507. Beginning of Combat Step",
    "508. Declare Attackers Step",
    "509. Declare Blockers Step",
    "510. Combat Damage Step",
    "511. End of Combat Step",
    "512. Ending Phase",
    "513. End Step",
    "514. Cleanup Step",
    # Section 6
    "600. General",
    "601. Casting Spells",
    "602. Activating Activated Abilities",
    "603. Handling Triggered Abilities",
    "604. Handling Static Abilities",
    "605. Mana Abilities",
    "606. Loyalty Abilities",
    "607. Linked Abilities",
    "608. Resolving Spells and Abilities",
    "609. Effects",
    "610. One-Shot Effects",
    "611. Continuous Effects",
    "612. Text-Changing Effects",
    "613. Interaction of Continuous Effects",
    "614. Replacement Effects",
    "615. Prevention Effects",
    "616. Interaction of Replacement and/or Prevention Effects",
    # Section 7
    "700. General",
    "701. Keyword Actions",
    "702. Keyword Abilities",
    "703. Turn-Based Actions",
    "704. State-Based Actions",
    "705. Flipping a Coin",
    "706. Rolling a Die",
    "707. Copying Objects",
    "708. Face-Down Spells and Permanents",
    "709. Split Cards",
    "710. Flip Cards",
    "711. Leveler Cards",
    "712. Double-Faced Cards",
    "713. Substitute Cards",
    "714. Saga Cards",
    "715. Adventurer Cards",
    "716. Class Cards",
    "717. Attraction Cards",
    "718. Prototype Cards",
    "719. Case Cards",
    "720. Omen Cards",
    "721. Controlling Another Player",
    "722. Ending Turns and Phases",
    "723. The Monarch",
    "724. The Initiative",
    "725. Restarting the Game",
    "726. Rad Counters",
    "727. Subgames",
    "728. Merging with Permanents",
    "729. Day and Night",
    "730. Taking Shortcuts",
    "731. Handling Illegal Actions",
    # Section 8
    "800. General",
    "801. Limited Range of Influence Option",
    "802. Attack Multiple Players Option",
    "803. Attack Left and Attack Right Options",
    "804. Deploy Creatures Option",
    "805. Shared Team Turns Option",
    "806. Free-for-All Variant",
    "807. Grand Melee Variant",
    "808. Team vs. Team Variant",
    "809. Emperor Variant",
    "810. Two-Headed Giant Variant",
    "811. Alternating Teams Variant",
    # Section 9
    "900. General",
    "901. Planechase",
    "902. Vanguard",
    "903. Commander",
    "904. Archenemy",
    "905. Conspiracy Draft"
]

def extract_metadata(chunkContent):
    # Split text into lines
    lines = chunkContent.split('\n')

    # Storage for results
    sections = []
    subsections = []
    rules = []
    subrules = []

    # Regular expressions for matching patterns at the start of lines
    # Sections and subsections: max 60 chars, must end with newline
    section_pattern = r'^(\d+)\.\s+(.{1,60})$'  # Matches "1. Game Concepts"
    subsection_pattern = r'^(\d{3,})\.\s+(.{1,60})$'  # Matches "100. General"
    rule_pattern = r'^(\d{3,}\.\d+)\.\s+(.+)$'  # Matches "100.1. ..."
    subrule_pattern = r'^(\d{3,}\.\d+[a-z])\s+(.+)$'  # Matches "100.1a ..."

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for subrule first (most specific)
        subrule_match = re.match(subrule_pattern, line)
        if subrule_match:
            subrules.append({
                'number': subrule_match.group(1),
                'text': subrule_match.group(2)
            })
            continue

        # Check for rule
        rule_match = re.match(rule_pattern, line)
        if rule_match:
            rules.append({
                'number': rule_match.group(1),
                'text': rule_match.group(2)
            })
            continue

        # Check for subsection
        subsection_match = re.match(subsection_pattern, line)
        if subsection_match:
            subsections.append({
                'number': subsection_match.group(1),
                'text': subsection_match.group(2)
            })
            continue

        # Check for section
        section_match = re.match(section_pattern, line)
        if section_match:
            sections.append({
                'number': section_match.group(1),
                'text': section_match.group(2)
            })

    return {
        'sections': sections,
        'subsections': subsections,
        'rules': rules,
        'subrules': subrules
    }


def ingestMTGRules(path, vectorDB):
  all_documents = []

  dir_path = Path(path)
  # Check if directory exists
  if not dir_path.exists():
    print(f"Directory {path} does not exist!!")

  # Find all PDF files in the directory
  pdf_files = list(dir_path.glob("*.pdf"))
  print(f"Found {len(pdf_files)} PDF files")

  # Load each PDF file
  for pdf_file in pdf_files:
    try:
      print(f"Loading: {pdf_file.name}")
      loader = PyPDFLoader(str(pdf_file), mode="single")
      documents = loader.load()
      all_documents.extend(documents)
      print(f"  - Loaded {len(documents)} pages from {pdf_file.name}")
    except Exception as e:
      print(f"  - Error loading {pdf_file.name}: {str(e)}")

  #document need to be splitted in smaller chunks
  chunk_size = 500
  chunk_overlap = chunk_size * 0.2
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = chunk_size, chunk_overlap = chunk_overlap
  )
  all_chunks = text_splitter.split_documents(all_documents)

  i=1
  last_section = ""
  last_subsection = ""
  last_rule = ""
  last_subrule = ""

  for chunk in all_chunks:
    #doc level metadata
    chunk.metadata["docType"] = "MTG Rules"
    #index
    chunk.metadata["index"] = i
    i += 1

    metadata = extract_metadata(chunk.page_content)

    sections_str = ' | '.join([f"{s['number']}. {s['text']}" for s in metadata['sections']
                          if f"{s['number']}. {s['text']}" in SECTIONS])
    if sections_str != "":
      last_section = sections_str.split(" | ")[-1]
    else:
      sections_str = last_section
    print(f"SECTIONS: {sections_str}")

    subsections_str = ' | '.join([f"{s['number']}. {s['text']}" for s in metadata['subsections']
                      if f"{s['number']}. {s['text']}" in SUBSECTIONS])
    if subsections_str != "":
      last_subsection = subsections_str.split(" | ")[-1]
    else:
      subsections_str = last_subsection
    print(f"SUBSECTIONS: {subsections_str}")

    rules_str = ' | '.join([f"{r['number']}" for r in metadata['rules']])
    if rules_str != "":
      last_rule = rules_str.split(" | ")[-1]
    else:
      rules_str = last_rule
    print(f"RULES: {rules_str}")

    subrules_str = ' | '.join([f"{sr['number']}" for sr in metadata['subrules']])
    if subrules_str != "":
      last_subrule = subrules_str.split(" | ")[-1]
    else:
      subrules_str = last_subrule
    print(f"SUBRULES: {subrules_str}")

    chunk.metadata["sections"] = sections_str
    chunk.metadata["subsections"] = subsections_str
    chunk.metadata["rules"] = rules_str
    chunk.metadata["subrules"] = subrules_str

    #Finally cleanup \n
    chunk.page_content = re.sub('\n', ' ', chunk.page_content)


  #print(all_chunks)
  vectorDB.add_documents(all_chunks)

  return

def metadata_func(record: dict, metadata: dict) -> dict:
    #Custom metadata extraction function to include specific card attributes.

    metadata["name"] = record.get("name", "")
    metadata["type_line"] = record.get("type_line", "")
    metadata["mana_cost"] = record.get("mana_cost", "")
    metadata["object_type"] = record.get("object", "")
    return metadata

def content_func(record: dict) -> str:
    #Custom content extraction function to create searchable text from card data.

    # Combine the key fields into a searchable text
    name = record.get("name", "")
    type_line = record.get("type_line", "")
    oracle_text = record.get("oracle_text", "")
    mana_cost = record.get("mana_cost", "")

    # Create a formatted content string
    content_parts = []
    if name:
        content_parts.append(f"Name: {name}")
    if type_line:
        content_parts.append(f"Type: {type_line}")
    if oracle_text:
        content_parts.append(f"Oracle Text: {oracle_text}")
    if mana_cost:
        content_parts.append(f"Mana Cost: {mana_cost}")

    return "\n".join(content_parts)

def ingestMTGCards(path, vector_db_Cards):
    #Load MTG card data from JSON file into Chroma vector database.

    all_documents = []

    dir_path = Path(path)
    # Check if directory exists
    if not dir_path.exists():
      print(f"Directory {path} does not exist!!")

    # Find all JSON files in the directory
    json_files = list(dir_path.glob("*.json"))
    print(f"Found {len(json_files)} PDF files")


    # Load each JSON file
    for json_file in json_files:
      try:
        print(f"Loading: {json_file.name}")

        # Process documents to use our custom content function
        processed_docs = []
        with open(str(json_file), 'r') as f:
            data = json.load(f)

        for item in data:
            if item.get("object") == "card":
                doc = Document(
                    page_content=content_func(item),
                    metadata={
                        "docType": "MTG Cards",
                        "name": item.get("name", ""),
                        "type_line": item.get("type_line", ""),
                        "mana_cost": item.get("mana_cost", ""),
                        "oracle_text": item.get("oracle_text", "")
                    }
                )
                processed_docs.append(doc)

        #print(processed_docs)
        vector_db_Cards.add_documents(processed_docs)
      except Exception as e:
        print(f"  - Error loading {json_file.name}: {str(e)}")
    return

def search_cards(vectorstore: Chroma, query: str, k: int = 5) -> List[Document]:
    #Search for cards in the vector store.
    results = vectorstore.similarity_search(query, k=k)
    return results

def transformQuery(user_query):
  # Query Transformation -> chamar uma LLM para transformar a pergunta do utilizador
  # HyDe
  prompt_query_transformation = f"""
    Instruction:
    Answer the provided user query with a hypothetical document that will be used in the similarity search.

    User Query:
    {user_query}
  """

  transformed_query = llm_query_transformation.invoke(prompt_query_transformation).content

  print(transformed_query)
  return transformed_query

def subQuery(user_query):
  # sub query
  prompt_sub_query = f"""
    Instruction:
    Descontruct the provided user query into 5 simple queries.
    For each simple query select what vector db collection I should search in between 'MTGRULES' (queries Magic The Gathering Rules) or 'MTGCARDS' (queries about existing MTG CARDS)

    User Query:
    {user_query}

    Return a json object with the following schema:
    {{
      queries: [{{collection: '', 'query': ''}}, ...]
    }}
  """

  llm_bind_json = llm_query_transformation.bind(response_format={"type": "json_object"})
  transformed_sub_queries = llm_bind_json.invoke(prompt_sub_query).content

  print(transformed_sub_queries)
  return transformed_sub_queries

def multiQuery(user_query):
  # multi query
  prompt_multi_query = f"""
    Instructions:
    Transform the given user query, into 5 different queries.

    User Query:
    {user_query}

    Return a json object with the following schema:
    {{
      queries: ['query_1', 'query_2', ...]
    }}
  """

  llm_bind_json = llm_query_transformation.bind(response_format={"type": "json_object"})
  transformed_sub_queries = llm_bind_json.invoke(prompt_multi_query).content

  print(transformed_sub_queries)
  return transformed_sub_queries


def inferencePhase(message, history):
  # Phase 2 - RAG Retrieval Augmentation
  user_query = message

  #transformed_query = transformQuery(user_query)
  #multi_queries = multiQuery(user_query)
  transformed_sub_queries= subQuery(user_query)

  queries = json.loads(transformed_sub_queries).get("queries")

  relevant_chunks_arr = []
  for query in queries:
    #print("Query: "+ query.get("query"))
    if query.get("collection") == "MTGRULES":
      relevant_chunks_arr.append(vector_db_MTG.similarity_search(query.get("query"), 3))
      #print("Chunks so far:")
      #print(relevant_chunks_arr)
    if query.get("collection") == "MTGCARDS":
      relevant_chunks_arr.append(vector_db_Cards.similarity_search(query.get("query"), 3))

  relevant_chunks_arr = np.array(relevant_chunks_arr).flatten()

  print(relevant_chunks_arr)

  string_of_chunks = ""
  for chunks in relevant_chunks_arr:
    if chunks.metadata["docType"] == "MTG Rules":
        string_of_chunks += "Extracted from Section: " + chunks.metadata['sections'] + " , Subsections: " + chunks.metadata['subsections'] + chunks.page_content + "\n\n"
    if chunks.metadata["docType"] == "MTG Cards":
        string_of_chunks += "Extracted from: " +  chunks.page_content + "\n\n"

  prompt = f"""
    User query:
    {user_query}

    Context:
    {string_of_chunks}
  """
  print(prompt)

  messages.append(("human", prompt))
  #print("ALL MESSAGES:")
  #print(messages)
  response = llm.invoke(messages)

  #keep this response for historical/memory purposes
  messages.append(("assistant", response.content))

  return response.content

#Remove comments in the first run to populate the vector DB
#ingestMTGRules(sourceFiles, vector_db_MTG)
#ingestMTGCards(sourceJson, vector_db_Cards)
gr.ChatInterface(inferencePhase).launch(debug=True)
