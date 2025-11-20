from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("Langchain_Document_Loaders/data.csv")

data = loader.load()

print(data[0])  

print(f"Total number of documents: {len(data)}")