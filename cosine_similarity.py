from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text=["London Paris London","Paris Paris London"]

#Used to count the frequency of the words in the text
cv=CountVectorizer()

count_matrix=cv.fit_transform(text)

#shows the count of words(London,Paris) in the text in an array format
print(count_matrix.toarray())

similarity_scores=cosine_similarity(count_matrix)

print(similarity_scores)
 
