import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


class tfidf_niche_finder(object):

    def __init__(self):

        self.custom_stop_words = [
            'logo',
            'shirt',
            'tshirt',
            'vintag',
            'retro',
            'movie',
            'officially',
            'licensed',
            'official',
            'offici',
            'licens',
            'graphic',
            'artwork',
            'gift',
            'men',
            'women'
        ]
    
    def load_documents(self, input_file):
        
        self.document_data_matrix = []
        self.document_data_by_asin = {}
        
        def _clean_document(document):
            
            document = document.lower()

            default_1 = "Lightweight, Classic fit, Double-needle sleeve and bottom hem".lower()
            default_2 = "Solid colors: 100% Cotton; Heather Grey: 90% Cotton, 10% Polyester; All Other Heathers: 50% Cotton, 50% Polyester Imported Machine wash cold with like colors, dry low heat ".lower()
            document = document.replace(default_1, '').replace(default_2,'')
            
            document.replace("tee shirt","")
            document.replace("t-shirt", "")
            document.replace(" t shirt", "")
            document.replace("shirt", "")
            document.replace("-", " ")
            
            document.replace("officially licensed","")
            
            if re.compile("officially licensed [\w\s]+ (apparel|shirt)").search(document):
                document = re.sub(r'officially licensed [\w\s]+ (apparel|shirt)', '', document)
            
            if re.compile("official [\w\s]+merchandise").search(document):
                document = re.sub(r'official [\w\s]+merchandise', '', document)

            if re.compile("graphic [\w\s\-]+shirt").search(document):
                document = re.sub(r'graphic [\w\s\-]+shirt', '', document)
            
            document = BeautifulSoup(document, 'html.parser').getText()
            
            return document

        with open(input_file, 'r') as f:
            for line in f:
                data = {}
                if len(line.replace("\"","").split('|')) == 11:
                    for item in line.replace("\"","").split('|'):
                        data[item.split(':',1)[0]] = item.split(':',1)[1]
                    data['document'] = _clean_document(data['title']) # + ". " + data['description'])

                    # Keep track of document data by asin
                    if data['asin'] not in self.document_data_by_asin:
                        self.document_data_by_asin[data['asin']] = data
                        self.document_data_matrix.append(data)

    def vectorize_documents(self):

        def tokenize_and_stem(text):
            # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                if not re.search('[^a-zA-Z]', token):
                    filtered_tokens.append(token)
            stems = [stemmer.stem(t) for t in filtered_tokens]
            return stems

        def tokenize_only(text):
            # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
            tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation) NOTE: this might be bad
            for token in tokens:
                if not re.search('[^a-zA-Z]', token):
                    filtered_tokens.append(token)
            return filtered_tokens
        
        stemmer = SnowballStemmer("english")
        
        stop_words = text.ENGLISH_STOP_WORDS.union(self.custom_stop_words)
        # print(stop_words)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.02, max_features=10000000,
                                           min_df=0, stop_words=stop_words,
                                           use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
        # print(tfidf_vectorizer.get_stop_words())

        self.tfidf_matrix = tfidf_vectorizer.fit_transform([document['document'] for document in self.document_data_matrix])
        # print(self.tfidf_matrix[0])
        # print(self.tfidf_matrix.shape)
        self.feature_names = tfidf_vectorizer.get_feature_names()


    def cluster_documents(self, similarity_threshold=0.2):
        
        self.niches = {}

        cosine_similarity_matrix = cosine_similarity(self.tfidf_matrix)

        for doc_vector_index, doc_vector in enumerate(cosine_similarity_matrix):

            self.niches[doc_vector_index] = {}
            self.niches[doc_vector_index]['similar_docs'] = set()
            self.niches[doc_vector_index]['percent_sales_ranks'] = 0.0
            self.niches[doc_vector_index]['average_sales_rank'] = "NA"
            self.niches[doc_vector_index]['sales_ranks'] = []
            self.niches[doc_vector_index]['hot'] = False
            self.niches[doc_vector_index]['consumed'] = False

            # Collect all similar documents
            for similar_doc_index, similar_doc in enumerate(doc_vector):
                if similar_doc > similarity_threshold:
                    self.niches[doc_vector_index]['similar_docs'].add(similar_doc_index)

            for similar_doc in self.niches[doc_vector_index]['similar_docs']:
            
                if doc_vector_index != similar_doc and similar_doc in self.niches and not self.niches[similar_doc]['consumed']:

                    intersecting_docs = niches[doc_vector_index]['similar_docs'].intersection(niches[similar_doc]['similar_docs'])
                    smaller_niche = min(len(niches[similar_doc]['similar_docs']),len(niches[doc_vector_index]['similar_docs']))                                                                          

                    # Check that the number of intersecting docs is greater than .75 the number of docs in the smaller niche
                    if (float(len(intersecting_docs))/float(smaller_niche)) > 0.75:
                        # print("these two share more than 75% of items")
                        # print(float(len(niches[doc_vector_index]['similar_docs'])))
                        # print(doc_vector_index)
                        # print(niches[doc_vector_index]['similar_docs'])
                        # print(document_data_matrix[doc_vector_index]['document'])
                        # print(float(len(niches[similar_doc])))
                        # print(similar_doc)
                        # print(niches[similar_doc]['similar_docs'])
                        # print(document_data_matrix[similar_doc]['document'])
                        # for item in niches[similar_doc]:
                        #     print(item)
                        #     print(document_data_matrix[item]['document'])

                        # Consume the smaller niche (probably want to consume larger niche instead)
                        if len(niches[doc_vector_index]['similar_docs']) < len(niches[similar_doc]['similar_docs']):
                            niches[doc_vector_index]['consumed'] = True
                        else:
                            niches[similar_doc]['consumed'] = True
    
    def get_hot_niches(self):
        
        def analyze_niche(niche):
            total_sales_rank = 0
            num_sales_ranks = 0
            for document in niche['similar_docs']:
                if self.document_data_matrix[document]['salesRank'] != "NA":
                    num_sales_ranks += 1
                    total_sales_rank += int(self.document_data_matrix[document]['salesRank'])
                    if 'best_sales_rank' in niche and int(self.document_data_matrix[document]['salesRank']) < int(niche['best_sales_rank']):
                        niche['best_sales_rank'] = self.document_data_matrix[document]['salesRank']
                    elif 'best_sales_rank' not in niche:
                        niche['best_sales_rank'] = self.document_data_matrix[document]['salesRank']
                        
            niche['percent_sales_ranks'] = float(num_sales_ranks) / float(len(niche['similar_docs']))
            if num_sales_ranks > 0:
                niche['average_sales_rank'] = float(total_sales_rank) / float(num_sales_ranks)

        def is_hot_niche(niche):
            if len(niche['similar_docs']) > 5:

                if niche['percent_sales_ranks'] > 0.8 and niche['average_sales_rank'] < 1500000:
                    return True
            
                if niche['percent_sales_ranks'] > 0.5 and niche['average_sales_rank'] < 900000:
                    return True
            
            return False

        num_hot_niches = 0
        for niche in niches:
            if len(niches[niche]['similar_docs']) > 0 and niches[niche]['consumed'] is False:
                analyze_niche(niches[niche])
                if hot_niche(niches[niche]):
                    num_hot_niches += 1
                    niches[niche]['hot'] = True

        print(num_hot_niches)
        with open("niches.txt", 'w') as f:
            for niche in niches:
                if niches[niche]['hot'] and not niches[niche]['consumed']:
                    f.write("number of documents in cluster: " + str(len(niches[niche]['similar_docs'])) + '\n')
                    f.write("percent sales ranks: " + str(round(niches[niche]['percent_sales_ranks'],2)) + "\n")
                    f.write("average sales rank: " + str(round(niches[niche]['average_sales_rank'],2)) + "\n")
                    f.write("best sales rank: " + str(niches[niche]['best_sales_rank']) + "\n")
                    for document in niches[niche]['similar_docs']:
                        f.write(document_data_matrix[document]['asin'] + ', ')
                    f.write("\n")
                    for document in niches[niche]['similar_docs']:
                        f.write(document_data_matrix[document]['document'] + "\n")
                    f.write("\n")

    def print_document_info(self, document_index):
        print("DOCUMENT INDEX {}: {}".format(document_index, self.document_data_matrix[document_index]['document']))

        if self.tfidf_matrix is not None and self.feature_names is not None:
            coo = self.tfidf_matrix[document_index].tocoo()
            for feature_index, feature in enumerate(coo.col):
                print("{} ({})".format(self.feature_names[feature], coo.data[feature_index]))

if __name__ == "__main__":

    niche_finder = tfidf_niche_finder()

    niche_finder.load_documents('shirts_newest_nt')
    print(niche_finder.document_data_matrix[0:1])

    niche_finder.vectorize_documents()
    niche_finder.print_document_info(0)

    # niche_finder.cluster_documents()

    # niche_finder.get_hot_niches()



