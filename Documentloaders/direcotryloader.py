from langchain_community.document_loaders import DirectoryLoader

loader=DirectoryLoader(
    path='books',
    glob='*.txt',
    show_progress=True,
    use_multithreading=True
)

docs=loader.load()

print(len(docs))
