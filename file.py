import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, UnidentifiedImageError
import pickle
import os
import pickle

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import VectorStore

class UploadedFile:
    def __init__(self, file) -> None:
        self.filename = "".join(file.name.split(".")[:-1])
        self.dir = f"data/{self.filename}"
        os.makedirs(self.dir, exist_ok=True)
        with open(f"{self.dir}/pitch.pdf", "wb") as f:
            f.write(file.read())

    def save_vector(self):
        self.create_images()
        self.extract_text()
        loader = TextLoader(f"{self.dir}/info.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key="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")
        vectors = FAISS.from_documents(documents, embeddings)

        with open(f"{self.dir}/vector.pkl", "wb") as f:
            pickle.dump(vectors, f)

    def get_vector(self):
        pickle_file = f"{self.dir}/vector.pkl"
        if not os.path.isfile(pickle_file):
            self.save_vector()

        with open(pickle_file, "rb") as f:
            vectors: VectorStore = pickle.load(f)
        return vectors


    def create_images(self):
        images = convert_from_path(f"{self.dir}/pitch.pdf")
        os.makedirs(f"{self.dir}/images", exist_ok=True)

        for idx, image in enumerate(images):
            file_path = f"{self.dir}/images/{idx}.ppm"
            with open(file_path, "wb") as file_object:
                image.save(file_object, "PPM")


    def extract_text(self):
        text_file = f"{self.dir}/info.txt"
        if os.path.exists(text_file):
            os.remove(text_file)

        for filename in os.listdir(f"{self.dir}/images/"):
            if filename.endswith(".ppm"):
                try:
                    with open(text_file, "a") as file:
                        image = Image.open(os.path.join(f"{self.dir}/images/", filename))
                        text = pytesseract.image_to_string(image)
                        file.write(text)
                except UnidentifiedImageError as e:
                    print(f"Error: {e}")
        return text_file
