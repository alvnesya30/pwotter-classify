import string
import nltk

tokenize = nltk.tokenize.word_tokenize
punctuation = string.punctuation

def main():
    import os
    import cv2
    import json
    import requests
    import numpy as np
    from bs4 import BeautifulSoup
    from matplotlib import pyplot as plt
    from nltk.probability import FreqDist
    from train import classifier

    # Create result directory if exists
    if os.path.isfile("result"): os.remove("result")
    if not os.path.isdir("result"): os.mkdir("result")
    
    # Install required package from NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Scrape Post from web
    URL = "https://academicslc.github.io/E222-COMP6683-YT01-00/"
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, "html.parser")
    # Get list items content
    card_post = soup.find("div", class_="card post")
    list_group = card_post.find("div", class_="list-group list-group-flush")
    list_items = list_group.find_all("div", class_="list-group-item d-flex")
    # Save response content (Optional)
    with open("result/response.html", "wb") as file:
        file.write(bytearray(str(list_items), "utf-8"))
    # Generate post data
    posts = []
    images = []
    for i in range(1, len(list_items)):
        post = {}
        list_item = list_items[i]
        container = list_item.find("div", class_="user-post-container")

        # Get source of profile image
        post["img"] = URL + list_item.find("img", class_="user-image")["src"]
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
        
        images.append(post["img"])
        posts.append(post)

    # Save post data in json (Optional)
    with open("result/response.json", "w") as out:
        json.dump({
            "data": posts
        }, out)

    # Image Preprocessing
    # Remove duplicates source on profile image
    images = list(set(images))
    
    def histogram(height, width, arr_img, index, color, label):
        # Get counter
        counter = np.zeros(256, dtype=int)
        for i in range(height):
            for j in range(width):
                counter[arr_img[i][j]] += 1
        # Plotting
        plt.subplot(1, 2, index)
        plt.plot(counter, color, label=label)
        plt.legend(loc='upper right')
        plt.ylabel('quantity')
        plt.xlabel('intensity')
    
    for img in images:
        # Get filename on image source
        filename = img.split('/')
        filename = filename[len(filename) - 1]
        # Get file image
        file = cv2.VideoCapture(img)
        file = file.read()[1]
        # BGR to GRAY
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        # Get Equelize file
        equ_file = cv2.equalizeHist(file)
        # Plotting histogram before and after equalize file
        plt.figure(filename, [16, 8])
        histogram(file.shape[0], file.shape[1], file, 1, "b", "Before")
        histogram(equ_file.shape[0], equ_file.shape[1], equ_file, 2, "r", "After")
        plt.savefig("result/figure_" + filename)
        plt.show()
        # Show image result before and after equalization
        image_result = np.hstack((file, equ_file))
        cv2.imshow(filename, image_result)
        cv2.imwrite("result/result_" + filename, image_result)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
