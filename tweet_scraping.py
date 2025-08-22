import tweepy
import json

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

tweets = api.search_tweets(q="Hadoop", count=99, lang="en")

with open("tweets.json", "w") as f:
    for tweet in tweets:
        json.dump(tweet._json, f)
        f.write("\n")