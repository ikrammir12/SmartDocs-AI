from unstructured.partition.pdf import partition_pdf

output_path = './content/'
file_path = output_path + 'attention.pdf'

#Extracting Data From PDF file

def partition_pdf_file(file_path):
    '''Partition a PDF file and return the elements.'''
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy='hi_res',
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
        chunking_strategy='by_title',
        max_characters=2000,
        combine_text_under_n_characters=500,
        new_after_n_characters=6000,

    )
    return chunks

chunks = partition_pdf_file(file_path)

#Processing The Extracted Data
def get_table(chunks):
    tables = []
    for chunk in chunks:
        for el in chunk.metadata.original_elements:
            if 'Table' in str(type(el)):
                print(el.to_dict())
                tables.append(el)

    return tables

tables = get_table(chunks) 


#Text and Images 

def save_texts(chunks):
    '''Extract text from the list of chunks.'''
    texts = [chunk for chunk in chunks if 'CompositeElement' in str(type(chunk))]
    return texts

texts = save_texts(chunks)

def get_image_base64(chunks):
    '''Get the base64 encoded images from the chunks.'''
    image_64 = []
    for chunk in chunks:
        chunk_el = chunk.metadata.original_elements
        for el in chunk_el:
            if 'Image' in str(type(el)):
                image_64.append(el.metadata.image_base64)

    return image_64

images = get_image_base64(chunks)

#Image and Text Summarization

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StructuredOutputParser

##Summarize Text
##create an openAi model 
from dotenv import load_dotenv
import os 
load_dotenv(override=True)
API_KEY = os.getenv('GEMINI_API_KEY')
model = ChatOpenAI(api_key = API_KEY , model='gemini-2.5-flash',base_url='https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.5-flash:generateContent', temperature=0.7)

##Create the prompt template for summarization
prompt = """
Your are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary ,no additional comment.
Do not start your message by saying 'Here is a summary' or anythings like that.
Just give the summary as it is.
Table or text chunk:{element}

"""


prompt_tamplate = ChatPromptTemplate.from_template(prompt)

#Chain the prompt with the gemini model and output parser
chain_text = prompt_tamplate|model|StructuredOutputParser

text_summaries = chain_text.batch(texts,{'max_concurrency':3})


#Image Summarization
prompt_template_image ="""

Describe the image in detail.
Be specific about the articture ,graphs,plots such as bar plot"""

messages =[
    (
        'user',
        [
            ##a text message containing our prompt instructions
            {'type':'text','text':prompt_template_image},
            ## an image message , passing the image as a base64-encoded URL
            {
            'type':'image_url',
            "image_url":{"url":"data:image/jpeg;base64,{image}"},                          
            },
        ],
    ),
]

##Create a ChatPromptTemplate Object form our structure messages
prompt_image = ChatPromptTemplate.from_messages(messages)

#chian the prompt with the gemini model and output parser
chain_image = prompt_image|model|StructuredOutputParser

image_summaries = chain_image.batch(images)


#Storing the summaries 
#As we generated the summaries now this time to store them into the vector database for future retrieval and using them

import uuid
import os 
from langchain_astradb import AstraDBVectorStore
from langchain_core.embeddings import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings

#Embedding model 
embedding_model = OpenAIEmbeddings(
    model = 'txt-embedding-004',
    api_key = API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

#text embeddings 
query_result = embedding.embed_query('hellow')
print(f'Query embedding: {query_result}')


#MultivectoreRetriever
from langchain.retrievers.multi_vector import MltiVectorRetriever

#Initialize the retriever
retriever = MltiVectorRetriever(
    vectorstore = vector_store,
    docstore = store,
    id_key = id_key
)

# We step our vectore store and document store , Now will load
#our text and image summaries

from langchain_core.documents import Document

#adding image and text summaries 

def loading_summaries(retriever,chunks,chunk_summary):
    """
        Generate ids for each chunk, create langchain document object for each summary chunk.
        Indexing the summary in vector store and document in docsotre.
    """
    ##Generate unique id for each chunk
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    ##creating the langchain document object for each text_summary chunk
    summary_texts = [Document(page_content=summary,metadata={id_key:doc_ids[i]}) for i,summary in enumerate(chunk_summary)]

    ##indexing the doumnets in vectore store and documents store
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids,chunks)))

####Adding text summaries to the vectore store and document store
loading_summaries_to_vector_store(retriever,texts,text_summaries)

##Adding imae summaries to vectore store and document store
loading_summaries_to_vector_store(retriever,images,image_summaries)
##now the retriver is to ready

#Creating The RAG chain
"""
with our data stored , We can now build the rag chain , This chain
will  orchestrate the entire process , from recining query to generating the response ."""
#Helper Funcation

from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode

def parse_docs(docs):
    """Parses the retrieved documents to reture a dictionary with text and images."""
    image_doc = []
    text_doc = []
    for doc in docs:
        try:
            ##Attempt to decode the document as an image
            b64decode(doc)
            ## If successful , appened to image_doc
            image_doc.append(doc)

        ## Raises binascii.Error if not a base64
        except Exception as e:
            ##if decoding fails , treat it as text 
            text_doc.append(doc)

    return {'images':image_doc,'text':text_doc}


def built_prompt(kwargs):
    """Builds the prompt for the LLM based on the model using the context and question."""

    ##extracting the context dictionary
    docs_by_type = kwargs['context']
    ##extracting the questions
    user_question = kwargs['question']

    ##if the length of text documents is greater than 0, concatenate the text
    context_text = ''
    if len(docs_by_type['text']) >0:
        for text_element in docs_by_type['text']:
            context_text += text_element.text


    ##Create a prompt with context including the images
    prompt_template = f"""
    Answer the question based only on the following context, which can include text and images.            
    Context : {context_text}
    Question : {user_question}""" 
    

    prompt_content = [{"type":"text","text":prompt_template}]

    ##if there are images , add them to the prompt 

    if len(docs_by_type['images']) >0:
        for image in docs_by_type['images']:
            prompt_content.append(
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image}"},
            }
            )

    ##return the prompt content
    return ChatPromptTemplate.from_messages(
        [HumanMessage(content=prompt_content)]
    )        


#Time to create the rage chain , in which i will combine the output from 
#model and Give the answer plus the documents that the model 
#used to generate the repsonse

from langchain_core.output_parsers import StrOutputParser

#the first chain gives a direct output

output_chain = (
    {
        ##Retriver will retrieve returns the retrieve document which we saw above when we invoked the retriever
        ##Then that will be passes to the parse_docs which will go over each document and parse it into text and images
        "context":retriever | RunnableLambda(parse_docs),
        "question":RunnablePassthrough(),
    }
    ##build_prompt will get the context which is a dictionary with text and images, and the question
    |RunnableLambda(built_prompt)
    |model
    |StrOutputParser()

)
##building the chain with the source reference 
# This will retunr a dictionary with the context , question, and the response from the model
chain_with_source = {
    'context':retriever | RunnableLambda(parse_docs)
    'question': RunnablePassthrough(),

} | RunnablePassthrough().assign(
    response = (
        RunnableLambda(built_prompt)
        |ChatOpenAI(model='gemini-2.5-flash')
        |StrOutputParser()
    )
)

#Generating a Response and Verifying the Source

import base64
from IPython.display import Image, display
def display_base64_image(base64_code):
    ##decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    #display the image
    display(Image(data=image_data))


#Assuming 'response' is the output of chain_with_source.invoke("Your question here")
response = chain_with_source.invoke('What is the summary of the document?')

##print the response and source documents
print('Response:',response['response'],"\n")
print("-"*80,"\n\nSource Douments:")

##print thesource text 
for text in response['context']['text']:
    print('page Number:',text.metadata.page_number)
    print(text.text)
    print("-"*80)

##source Images
for image in response['context']['image']:
    display_base64_image(image)   
 
            
