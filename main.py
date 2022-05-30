import requests
import string
import nltk 

tokenize = nltk.tokenize.word_tokenize
punctuation = string.punctuation 

def main():
    import json
    from bs4 import BeautifulSoup
    from nltk.probability import FreqDist
    from train import classifier
    
    # Install required package from NLTK 
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Scrape Post from web
    URL = "https://academicslc.github.io/E222-COMP6683-YT01-00/"
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, "html.parser")
    card_post = soup.find("div", class_="card post")
    list_group = card_post.find("div", class_="list-group list-group-flush")
    list_items = list_group.find_all("div", class_="list-group-item d-flex")
    # Optional Command
    with open("response.html", "wb") as file:
        file.write(bytearray(str(list_items), "utf-8"))
    # Generate post data
    posts = []
    for i in range(1, len(list_items)):
        post = {}
        list_item = list_items[i]
        container = list_item.find("div", class_="user-post-container")
        
        # Get and lower case post content
        post["post_content"] = container.find("div", class_="user-post-content").text.lower()
        # Remove Punctuation
        post["post_content"] = "".join(word for word in post["post_content"] if word not in punctuation)
        # Tokenize
        post["post_content"] = tokenize(post["post_content"])
        # Classify
        result_classify = classifier.classify(FreqDist(post["post_content"]))
        # Display result
        print(f"Post {i} : {result_classify} post")
        
        # Get profile image
        post["img"] = URL + list_item.find("img", class_="user-image")["src"]
        
        posts.append(post)

    # Optional Command
    with open("response.json", "w") as out:
        json.dump({
            "data": posts
        }, out)

if __name__ == '__main__':
    main()
