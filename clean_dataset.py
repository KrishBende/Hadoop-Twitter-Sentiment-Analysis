import pandas as pd

df=pd.read_csv("customer_service_kaggle_dataset.csv")
print(df.columns)

new_df=df.drop(['author_id','inbound','created_at','response_tweet_id', 'in_response_to_tweet_id'],axis=1)

print(new_df)

new_df.to_csv("tweets.csv",header=False)