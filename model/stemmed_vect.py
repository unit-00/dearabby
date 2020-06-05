import nltk
nltk.download('stopwords')


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from nltk.stem import SnowballStemmer
    
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer('english')
        analyzer = super().build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc) if w.isalpha() and len(w) > 3 and w not in {'abby', 'year', 'time'}])
    
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer('english')
        analyzer = super().build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc) if w.isalpha() and len(w) > 3 and w not in {'abby', 'year', 'time'}])
    
class StemmedHashingVectorizer(HashingVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer('english')
        analyzer = super().build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc) if w.isalpha() and len(w) > 3 and w not in {'abby', 'year', 'time'}])
