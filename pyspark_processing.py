#imports
from pyspark.sql import SparkSession
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF

#Initial fetching
spark = SparkSession.builder \
    .appName("TwitterSentimentAnalysis") \
    .getOrCreate()

df = spark.read.csv("hdfs://localhost:9000/twitter_data/tweets.csv")
df= df.withColumnRenamed("_c0", "index") \
        .withColumnRenamed("_c1", "tweet_id") \
        .withColumnRenamed("_c2", "tweet_text")

df = df.filter(df.tweet_text.isNotNull())
               
print("First 5 records")
df.show(5, truncate=False)

# Define UDF to compute sentiment polarity
def vader_score(text):    
    vader_sentiment = SentimentIntensityAnalyzer()
    score = vader_sentiment.polarity_scores(text) 
    return score['compound']

vader_sentiment_udf = F.udf(vader_score, FloatType())

def blob_sentiment(text):
    return TextBlob(text).sentiment.polarity

blob_sentiment_udf = F.udf(blob_sentiment, FloatType())

# Add a sentiment column
df = df.withColumn("vader_sentiment", vader_sentiment_udf(F.col("tweet_text")))
df = df.withColumn("blob_sentiment", blob_sentiment_udf(F.col("tweet_text")))

# Label positive, neutral, negative
df = df.withColumn(
    "vader_sentiment_label",
    F.when(df.vader_sentiment > 0, "positive")
     .when(df.vader_sentiment < 0, "negative")
     .otherwise("neutral")
)

df = df.withColumn(
    "blob_sentiment_label",
    F.when(df.blob_sentiment > 0, "positive")
     .when(df.blob_sentiment < 0, "negative")
     .otherwise("neutral")
)

print("First 5 records after sentiment analysis")
df.select("tweet_text", "vader_sentiment", "vader_sentiment_label", "blob_sentiment", "blob_sentiment_label").show(5, truncate=False)

#Sentiment Analysis
df.groupBy("blob_sentiment_label").count().show()
df.groupBy("vader_sentiment_label").count().show()

df = df.withColumn(
    "agreement",
    F.when(F.col("blob_sentiment_label") == F.col("vader_sentiment_label"), "agree").otherwise("disagree")
)
df.groupBy("agreement").count().show()

#Comparison Visualisation
pdf_vader = df.groupBy("vader_sentiment_label").count().toPandas()
pdf_vader = pdf_vader.sort_values("vader_sentiment_label")
sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
sns.barplot(x='vader_sentiment_label', y='count', data=pdf_vader, palette=["green", "red", "gray"])
plt.title("Sentiment Distribution (Vader)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()

pdf_blob = df.groupBy("blob_sentiment_label").count().toPandas()
pdf_blob = pdf_blob.sort_values("blob_sentiment_label")
plt.figure(figsize=(6,4))
sns.barplot(x='blob_sentiment_label', y='count', data=pdf_blob, palette=["green", "red", "gray"])
plt.title("Sentiment Distribution (BLOB)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()

#WordClouds

df_clean = df.withColumn("clean_text", F.regexp_replace(F.lower(F.col("tweet_text")), "[^a-zA-Z\\s]", ""))
df_pos = df_clean.filter(F.col("vader_sentiment_label") == "positive")
df_neg = df_clean.filter(F.col("vader_sentiment_label") == "negative")

pos_words = " ".join([row.clean_text for row in df_pos.select("clean_text").collect()])
neg_words = " ".join([row.clean_text for row in df_neg.select("clean_text").collect()])

wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(pos_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Tweets WordCloud")
plt.show()

wordcloud_neg = WordCloud(width=800, height=400, background_color="white").generate(neg_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Tweets WordCloud")
plt.show()


#Compute TF-IDF or frequent words with PySpark MLlib
# Tokenize
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
df_words = tokenizer.transform(df_clean)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_words = remover.transform(df_words)

# CountVectorizer
cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
cv_model = cv.fit(df_words)
df_words = cv_model.transform(df_words)

# IDF
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df_words)
df_words = idf_model.transform(df_words)

# Show top words for first row (optional)
vocab = cv_model.vocabulary
first_row = df_words.select("tfidf_features").first()
tfidf_scores = first_row["tfidf_features"]
top_words = sorted(zip(tfidf_scores.indices, tfidf_scores.values), key=lambda x: x[1], reverse=True)
for idx, score in top_words[:10]:
    print(vocab[idx], score)


#Save Final CSV
df.write.csv("hdfs://localhost:9000/twitter_data/output_sentiments", header=True, mode="overwrite")

# Stop Spark session
spark.stop()