from app.ui import page_intro


# page1_title = "HOME page"
page1_content = """
----

This is Home page

... SOME CONTENT ...

-----
"""

@page_intro(page1_content)  
def run():
    pass

run()