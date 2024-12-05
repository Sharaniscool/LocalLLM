from unstructured.partition.pdf import partition_pdf
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
import re

from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from langchain_community.embeddings.ollama import OllamaEmbeddings
from tqdm import tqdm
import uuid
import re

from sentence_transformers import CrossEncoder

import pandas as pd

client = QdrantClient(url="http://localhost:6333")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

maxTokens = 500
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, trim=False, capacity=maxTokens)

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def filterExcel(xlsxFile):
    df = pd.read_excel(xlsxFile,sheet_name=1)
    filtered_df = df.iloc[:,[3,4,5,6]]
    return filtered_df



def extractTOC(sepData):
    sepData = [
    element for element in sepData
    if "unstructured.documents.elements.Image" not in str(type(element))
    ]
    # Initialize a list to store pages with tables following a Table of Contents
    toc_following_tables = []

    # Iterate over the elements to find "Table of Contents"
    for i, element in enumerate(sepData):
        # Check for Table of Contents
        if "Table of Contents" in str(element):
            current_page_number = getattr(element.metadata, 'page_number', None)
            # Collect all elements on the same page as the Table of Contents
            toc_page_elements = [
                str(elem) for elem in sepData
                if getattr(elem.metadata, 'page_number', None) == current_page_number
            ]
            
            # Check if the next page has any table elements
            next_page_number = current_page_number + 1
            next_page_text = []
            for next_element in sepData[i+1:]:
                if getattr(next_element.metadata, 'page_number', None) == next_page_number:
                    if type(next_element).__name__ == 'Table':
                        # Collect text from the next page
                        next_page_text = [
                            str(elem) for elem in sepData
                            if getattr(elem.metadata, 'page_number', None) == next_page_number
                        ]
                        toc_following_tables.append((current_page_number, toc_page_elements, next_page_number, next_page_text))
                        break

    # Write the text from pages with tables following a Table of Contents to separate text files
    for toc_page, toc_text, table_page, table_text in toc_following_tables:
        toc_filename = f"C:/sharan/ragllm/unstruc1/test_app/testData/toc_page_{toc_page}.txt"
        table_filename = f"C:/sharan/ragllm/unstruc1/test_app/testData/table_page_{table_page}.txt"
        
        with open(toc_filename, "w") as toc_file:
            # toc_file.write(f"TOC on page {toc_page}\n")
            toc_file.write("\n".join(toc_text) + "\n")
        
        with open(table_filename, "w") as table_file:
            # table_file.write(f"Table on page {table_page}\n")
            table_file.write("\n".join(table_text) + "\n")





def clean_trailing_dots(content):
    sample_string = content
    # Regular expression to remove trailing dots from each segment
    cleaned_string = re.sub(r'\.{2,}', '', sample_string)

    # Print the cleaned string
    print(cleaned_string)
    return cleaned_string



def structureLLM(context):

    instruction = (   "Your task is to identify and provide the title and page number for each entry in the provided context. "
        "You have to create a python list of lists in which the first element of the list is the section number and the second element is the page number"
        "your format should be like this [['Title',Y],['Title',Y]] where Y is the Page number"
        "Your response should be in the following format '''[<list of content>]'''"
   )
    
    prompt_template = (
        f"{instruction}\n\n"
        f"Context:table of contents:\n{context}\n\n"
        # f"Question: {question}\n\n"
        f"Answer:"
    )
    # model = Ollama(model="phi3.5")
    model = Ollama(model="mistral:latest")
    # model = Ollama(model="phi3:mini")

    response = model.invoke(prompt_template)
    
    print(f"\nModel Response:\n{response}\n")
    return response



def parseIndex(file_path):
    index_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by colon
            parts = line.split(':')
            if len(parts) < 3:
                # Skip lines that don't have the expected format
                continue

            section_number = parts[0].strip()
            title = parts[1].strip()
            page_part = parts[2].strip()

            # Ensure 'Page' is in the page_part
            if 'Page' in page_part:
                page = page_part.split('Page')[1].strip()

                # Add to dictionary
                if page not in index_dict:
                    index_dict[page] = []
                index_dict[page].append([section_number, title])

    return index_dict



def extractParentChunks(sepData, index_dict):
    extracted_sections = []
    extracted_texts = {}
    not_found_titles = []

    # Flatten the index_dict to get a list of all titles with their page numbers
    all_titles = [(int(page), section_number, title) for page, sections in index_dict.items() for section_number, title in sections]

    for i, (current_page, section_number, title) in enumerate(all_titles):
        # Gather all text from the current page
        page_data = [element.text for element in sepData if element.metadata.page_number == current_page]
        combined_text = " ".join(page_data)

        found = False
        if title in combined_text:
            found = True
            start_index = combined_text.index(title)
            
            # Look ahead in the next 5 pages to find the next title
            lookahead_text = combined_text
            for offset in range(1, 6):
                next_page_data = [element.text for element in sepData if element.metadata.page_number == current_page + offset]
                lookahead_text += " " + " ".join(next_page_data)
            
            # Find the next title
            end_index = len(lookahead_text)
            next_title_found = False
            for j in range(i + 1, len(all_titles)):
                next_page, next_section_number, next_title = all_titles[j]
                if next_title in lookahead_text[start_index:]:
                    end_index = lookahead_text.index(next_title, start_index)
                    next_title_found = True
                    break
            
            # If the next title is not found within the lookahead, use the page number of the next title
            if not next_title_found and i + 1 < len(all_titles):
                next_page, _, _ = all_titles[i + 1]
                # Extract all elements between current title's page and next title's page
                extended_text = lookahead_text
                for offset in range(current_page + 1, next_page):
                    extended_page_data = [element.text for element in sepData if element.metadata.page_number == offset]
                    extended_text += " " + " ".join(extended_page_data)
                end_index = len(extended_text)
                extracted_texts[current_page] = extended_text[start_index:end_index].strip()
            else:
                extracted_texts[current_page] = lookahead_text[start_index:end_index].strip()

            extracted_sections.append(title)
        else:
            not_found_titles.append(title)

    return extracted_sections, extracted_texts, not_found_titles




def extractChildChunks(extracted_texts):
    semantic_chunks = {}
    all_chunks = []

    for page_number, text in extracted_texts.items():
        # Use the splitter to create semantic chunks
        chunks = splitter.chunks(text)
        # Store the chunks in the dictionary with the page number as the key
        semantic_chunks[page_number] = chunks
        # Add chunks to the all_chunks list
        all_chunks.extend(chunks)

    return semantic_chunks, all_chunks



def storeParentChunks(extracted_texts):
    # Define your collection name
    collection_name = "Parent_Chunks_Test"

    # Check if the collection exists, and create it if it doesn't
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance="Cosine")
        )

    # Assuming extracted_texts is a dictionary with page numbers as keys and text as values
    for page_number, text in tqdm(extracted_texts.items(), desc="Loading text chunks"):
        # Generate the embedding for the text
        vector = embeddings.embed_query(text)
        # Create a PointStruct with the page number as the ID, vector, and payload
        point = PointStruct(id=page_number, vector=vector, payload={"text": text})
        # Upsert the vector into the Qdrant collection
        client.upsert(collection_name=collection_name, points=[point])

def storeChildChunks(semantic_chunks):
    collection_name = "Child_chunks_Test"

    # Check if the collection exists, and create it if it doesn't
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance="Cosine")
        )

    # Assuming semantic_chunks is a dictionary with page numbers as keys and lists of chunks as values
    for page_number, chunks in tqdm(semantic_chunks.items(), desc="Loading semantic chunks"):
        for chunk in chunks:
            # Generate the embedding for the chunk
            vector = embeddings.embed_query(chunk)
            # Create a PointStruct with a UUID as the ID, vector, and payload
            point = PointStruct(
                id=str(uuid.uuid4()), 
                vector=vector, 
                payload={"text": chunk, "page_number": page_number}
            )
            # Upsert the vector into the Qdrant collection
            client.upsert(collection_name=collection_name, points=[point])



def rerank_hits(query, hits):
    """Reranks hits based on semantic similarity to the query."""
    sentence_pairs = [[query, hit.payload['text']] for hit in hits]
    similarity_scores = cross_encoder_model.predict(sentence_pairs)
    scored_hits = list(zip(hits, similarity_scores, [hit.payload['page_number'] for hit in hits]))
    scored_hits.sort(key=lambda x: x[1], reverse=True)
    return scored_hits

def query_and_retrieve_parent(query):
    """Queries the child chunks collection, reranks hits, and retrieves the parent chunk."""
    # Query the child chunks collection
    search_results = client.search(
        # collection_name="Child_chunks",
        collection_name="Child_chunks_Test",
        query_vector=embeddings.embed_query(query),
        limit=1  # Adjust the limit as needed
    )
    print(f"Child Chunks: {search_results}")
    # Re-rank the search results
    scored_hits = rerank_hits(query, search_results)

    # Extract the page number of the top-ranked chunk
    top_chunk_page_number = scored_hits[0][2]

    # Retrieve the parent chunk using the page number as the ID
    parent_chunk_response = client.retrieve(
        collection_name="Parent_Chunks_Test",
        # collection_name="Parent_Chunks",
        ids=[top_chunk_page_number]
    )
    # parent_chunk_response = parent_chunk_response.payload['text']
    return parent_chunk_response , top_chunk_page_number

def agentLLM(context, question):
    """Interacts with the LLM to generate a response based on the context."""
    instruction = ("""You are part of the RAG LLM process. Your task is to identify P&ID numbers mentioned in the given context and create a Python list of these numbers. 
                   You do not need to prepend any additional text to the P&ID numbers. 
                   Format your response as follows:
                   ```
                   PID_number = [<list_of_pid_numbers>]
                   ```
                   """)
    context = context[:250]
    prompt_template = (
        f"Instruction: {instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    print(prompt_template)
    model = Ollama(model="mistral:latest")
    response = model.invoke(prompt_template)
    return response

# Example usage
# query = "what are the H-1106 Feed Control Field devices"
# parent_chunk = query_and_retrieve_parent(query)
# print("Parent Chunk:", parent_chunk)

# if parent_chunk:
#     context = parent_chunk[0].payload['text']
#     response = agentLLM(context, query)
#     print("LLM Response:", response)
# else:
#     print("Parent chunk not found.")


def getData(xlsxFile,sepData):
    print("inside getData")
    filtered_df = filterExcel(xlsxFile)
    # print("extracting toc")
    # extractTOC(sepData)

    # with open("C:/sharan/ragllm/unstruc1/test_app/testData/toc_page_2.txt", "r",encoding="utf-8") as file:
    #     content1 = file.read()

    # with open("C:/sharan/ragllm/unstruc1/test_app/testData/table_page_3.txt", "r",encoding="utf-8") as file:
    #     content2 = file.read()

    # print("cleaning trailing dots")
    # content1= clean_trailing_dots(content1)
    # content2= clean_trailing_dots(content2)

    # # list1 = structureLLM(content1)
    # # list2 = structureLLM(content2)
    # print("######################################")
    # # print(list1+list2)
    # print("######################################")

    # file_path = 'C:/sharan/ragllm/unstruc1/test_app/testData/wholeIndex.txt'
    # index_dict = parseIndex(file_path)

    # extracted_sections, extracted_texts, not_found_titles = extractParentChunks(sepData, index_dict)
    # print("Extracted Sections:", extracted_sections)
    # print("Extracted Texts:", extracted_texts)
    # print("Not Found Titles:", not_found_titles)

    # semantic_chunks, all_chunks = extractChildChunks(extracted_texts)
    # print("Semantic Chunks by Page:", semantic_chunks)
    # print("All Semantic Chunks:", all_chunks)

    # storeParentChunks(extracted_texts)
    # storeChildChunks(semantic_chunks)
    # print("DONE")
    return filtered_df

def excelData(query: str,filtered_df: object):
    parent_chunk ,page_number= query_and_retrieve_parent(query)
    print("Parent Chunk:", parent_chunk)

    if parent_chunk:
        context = parent_chunk[0].payload['text']
        response = agentLLM(context, query)
        print("LLM Response:", response)
    else:
        print("Parent chunk not found.")
        return "No available data found use context"

    response = response.replace("`", "")
    response = response.strip()
    print(response)

    try:
        print("Executing response:")
        # Create a dictionary to capture the local variables
        local_vars = {}
        exec(response, {}, local_vars)  # Execute the code contained in the response

        # Access PID_number from the local_vars dictionary
        PID_number = local_vars.get('PID_number', None)
        if PID_number is None:
            raise ValueError("PID_number is not defined in the executed code.")

        print(f"PID_number: {PID_number}")
        print(type(PID_number))
        pattern = '|'.join(PID_number)
        print(f"Pattern: {pattern}")

        # Filter the DataFrame
        filtered_df = filtered_df[filtered_df['P&ID'].str.contains(pattern)]
        filtered_df = filtered_df.iloc[:8, :]
        return filtered_df.to_string()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "No available data found use context"



