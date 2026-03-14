def build_retriever(vector_store, k=4, fetch_k=12):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
