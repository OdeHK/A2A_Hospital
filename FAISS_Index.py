import numpy as np
from langchain_community.vectorstores import FAISS
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document

def FAISS_CS(data, embeddings):

    vectors = embeddings.embed_documents(data)
    vectors = np.array(vectors,dtype='float32')
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Tạo Document cho từng câu
    documents = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(data)]

    # Tạo docstore và mapping index → docstore_id
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # 6. Gói thành FAISS vectorstore (dùng cho LangChain retriever)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    return vectorstore




