import pickle

classifier = None

def create_model():
    """
    Create model.pickle 
    """
    print("Creating model data!")
    import nltk
    from nltk.classify import NaiveBayesClassifier, accuracy
    from main import punctuation
    
    import time
    start_time = time.time()

    # Get package from NLTK Corpus
    # Install required package on nltk 
    try:
        nltk.data.find("corpora/movie_reviews")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("movie_reviews")
        nltk.download("stopwords")
    from nltk.corpus import movie_reviews, stopwords

    # # Optional command for experimental
    # # Get several file positive and negative reviews
    # fileids_current = movie_reviews.fileids()
    # fileids = []
    # index = 1
    # clear_index = 1
    # while (clear_index < 3):
    #     if (index > 12):
    #         index = 1
    #         clear_index += 1
    #         fileids_current = list(reversed(fileids_current))
    #         continue
    #     fileids.append(fileids_current[index - 1])
    #     index += 1

    # Get word features and documents
    print("Getting word features and documents..")
    word_features = []
    documents = []
    label_filter = {
        "pos": "positive",
        "neg": "negative",
    }
    for paragraph in movie_reviews.fileids(): # Change this array variable with fileids if use Optional Command on line 27-40
        words = movie_reviews.words(paragraph)
        # Clean punctuations and lower case
        words = [word.lower() for word in words if word not in punctuation]
        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]
        # Add words to word features
        word_features += words
        # Add words to document tuple with label
        documents.append((words, label_filter[paragraph.split('/')[0]]))
    # Remove duplicate word on word_features
    word_features = list(set(word_features))
    # Generate datasets from documents
    print("Generating datasets..")
    featuresets = []
    for sentences, label in documents:
        features = {}
        for word in word_features:
            features[word] = word in sentences
        featuresets.append((features, label))
    import random
    random.shuffle(featuresets)
    # Train model data
    print("Training model data..")
    train_count = int(len(featuresets) * 0.9)
    train_data = featuresets[:train_count]
    test_data = featuresets[train_count:]
    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy:", accuracy(classifier, test_data) * 100)
    # Save model data
    print("Saving model data..")
    with open("model.pickle", "wb") as file:
        pickle.dump(classifier, file)

    print("Model data created!")
    end_time = time.time()
    print("Time:", end_time - start_time, 's')
    
    return classifier


if __name__ == '__main__':
    create_model()
else:
    try:
        with open("model.pickle", 'rb') as file:
            classifier = pickle.load(file)
    except:
        print("Failed to load model data!")
        classifier = create_model()
