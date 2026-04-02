from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.lazy_load()

for documents in docs:
    print(documents.metadata)

# print(docs[0].page_content)
# print(docs[1].metadata)