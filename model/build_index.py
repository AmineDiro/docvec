import wikipediaapi

wiki_html = wikipediaapi.Wikipedia(
    user_agent="MyProjectName (merlin@example.com)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.HTML,
)
p_html = wiki_html.page("Python_(programming_language)")

with open("python_wiki.html", "w") as f:
    f.write(p_html.text)
