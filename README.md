# mtg-rag-chatbot
Simple RAG chatbot to assist users in understanding the complex rules of MTG in a more digestable way.
The project consists in a simple RAG to assist users in understanding the complex rules of MTG in a more digestable way.

RAG (Retrieval Augmented Generation) is a technique that encapsulates all the previous topics with the final task for augmenting LLM knowledge with additional data.
Architecture:
	Phase 1 - Indexing:
		Load: First we need to load our data. This is done with Document Loaders.
		Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won’t fit in a model’s finite context window.
		Store: We need somewhere to store and index our splits, so that they can later be searched over. This is often done using a VectorStore and Embeddings model.

	Phase 2 - RETRIEVAL AND GENERATION
		Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
		Generate: A LLM produces an answer using a prompt that includes the question and the retrieved data (Context)

Use Case: Assisting users in understanding the complex rules of a game (MTG) in a more digestable way. Assist in finding Cards.
Background: Magic: The Gathering (MTG) is often refered as the most complex game in the world (reference: https://www.technologyreview.com/2019/05/07/135482/magic-the-gathering-is-officially-the-worlds-most-complex-game/)
and is hard to keep track of all rules, layers, interactions and game states.
Proposed solution: Ingest the game rules (pdf document with over 280 pages) and all cards printed so far (JSON file) to retrieve relevant context enhancing the quality of reply from the a pre-trained LLM.

Tech decisions:
	Coded in Python
	OPENAI api
	Chroma Vector DB
	Langchain_community document loaders and text splitters
	LLMs: gpt-4o and gpt-4.1-nano
	Embedding model: text-embedding-3-small
	Web UI for LLM: Gradio

Functional decisions:
	1. Rules and Cards indexing are stored in separate collections
	2. Improvement Strategies:
		Ingestion stage:
			Extract relevant Metadata from each document:
				MTG Rules: Section, Subsection, Rule and Subrule
				MTG Cards: Name, Cost, Card Type and Oracle text
		Context retrieving stage:
			Divide user query into shorter, simpler versions and deduct if its a question about Rules or Cards
		Answer generation:
			Use original user prompt, retrieved context and chat history (past questions and answers) to generate the answer.			
