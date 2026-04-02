from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('dl-curriculum.pdf')
docs=loader.load()

# text = """
# A standard 300-word essay typically spans 0.6 pages single-spaced or 1.2 pages double-spaced when using 12-point Times New Roman or Arial font with 1-inch margins.  This length generally consists of 5 to 6 paragraphs (one introduction, three body paragraphs, and one conclusion) and takes approximately one minute to read.

# To write a 300-word piece effectively, you should follow a concise structure:

# Introduction (50-70 words): Start with a hook, provide background, and state your thesis. 
# Body (100-150 words): Develop your main points with evidence, keeping each paragraph focused on a single idea. 
# Conclusion (30-50 words): Summarize your key points and restate the thesis without introducing new information. 
# Common examples of 300-word texts include academic essays, short blog posts, and news articles. Specific samples often explore topics like career goals, personal narratives (such as "Who am I?"), or brief analyses of subjects like Romeo and Juliet. While 300 words is a common minimum for content to avoid being flagged as "thin" in some contexts, SEO guidelines suggest that 500 words is often the ideal target for better search engine performance.

# """

split=CharacterTextSplitter(chunk_overlap=0,chunk_size=150,separator='')
result=split.split_documents(docs)
print(result[0].page_content)