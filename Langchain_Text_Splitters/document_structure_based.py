from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

loader = PyMuPDFLoader("Langchain_Text_Splitters/YogaforMentalHealth.pdf")
documents = loader.lazy_load()

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=500,
    chunk_overlap=100,
    language=Language.MARKDOWN,
)

result = text_splitter.split_documents(documents)

print(f"Number of documents: {len(result)}")
print(f"First document: {result[20].page_content}")
print(f"Second document: {result[21].page_content}")


Text = """
Class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def study(self):
        print(f"{self.name} is studying.")

    def sleep(self):
        print(f"{self.name} is sleeping.")

    def attend_class(self, subject):
        print(f"{self.name} is attending {subject} class.")

    def take_exam(self, subject):
        print(f"{self.name} is taking an exam in {subject}.")

# Example usage
student1 = Student("Alice", 20)
student1.study()
student1.attend_class("Mathematics")
student1.take_exam("Mathematics")
student1.sleep()
"""

# Create a Document object directly from the text instead of using TextLoader
documents1 = [Document(page_content=Text)]

text_splitter1 = RecursiveCharacterTextSplitter.from_language(
    chunk_size=100,
    chunk_overlap=20,
    language=Language.PYTHON,
)

result1 = text_splitter1.split_documents(documents1)

print(f"Number of documents: {len(result1)}")
print(f"First Python document: {result1[0].page_content}")
print(f"Second Python document: {result1[1].page_content}")