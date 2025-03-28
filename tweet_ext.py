import twitter
import urllib.request
import os
import config as cf
import pickle
import pandas as pd


#Invoking Twitter API
api = twitter.Api(consumer_key = cf.credentials["consumer_key"],
                  consumer_secret = cf.credentials["consumer_secret"],
                  access_token_key = cf.credentials["access_token"],
                  access_token_secret = cf.credentials["access_token_secret"])



hashtag = 'flood'
result_type = 'mixed' # possible values: mixed, recent, popular
include_entities = 'true'
with_twitter_user_id = 'true' # include user information
since = '2021-07-20' # start date
until = '2021-07-23'
count = '50' # The number of tweets to return per page



query = ('q={hashtag}' + 
         '&result_type={result_type}' +
         '&include_entities={include_entities}' +
         '&with_twitter_user_id={with_twitter_user_id}' + 
         '&since={since}' + 
         '&until={until}' +
         '&count={count}')

query = query.format(hashtag=hashtag,
                 result_type=result_type,
                 include_entities=include_entities,
                 with_twitter_user_id=with_twitter_user_id,
                 since=since,
                 until=until,
                 count=count)



all_results = []
max_id = None
IDs = []

for i in range(0,180):
    
    results = api.GetSearch(raw_query = query)
    all_results.extend(results)
    IDs = [result.id for result in results]
    smallest_ID = min(IDs)
    
    if max_id == None: # first call 
        max_id = smallest_ID
        query += '&max_id={max_id}'.format(max_id=max_id)
    else:
        old_max_id = "max_id={max_id}".format(max_id=max_id)
        max_id = smallest_ID
        new_max_id = "max_id={max_id}".format(max_id=max_id)
        query = query.replace(old_max_id,new_max_id)



sample_result = all_results[1]

tweet_id = sample_result.id_str
screen_name = sample_result.user.screen_name

tweet_url = "https://twitter.com/{screen_name}/status/{tweet_id}"
tweet_url = tweet_url.format(screen_name=screen_name,tweet_id=tweet_id)


def get_tweet_url(tweet):
    
    tweet_id = tweet.id_str
    screen_name = tweet.user.screen_name

    tweet_url = "https://twitter.com/{screen_name}/status/{tweet_id}"
    tweet_url = tweet_url.format(screen_name=screen_name,tweet_id=tweet_id)
    
    return tweet_url



folder_name = "downloaded_media2"


image_origins = {
    "tweet_url": [],
    "image_id": [],
    "image_url": [],
}

downloaded_img_ids = [file[:file.find('.')] for file in os.listdir(folder_name)]

for tweet in all_results:
    
    tweet_url = get_tweet_url(tweet)
    
    if tweet.media:
        
        for media in tweet.media:
            
            media_id = str(media.id)
            media_url = media.media_url
            
            if not(media_id in downloaded_img_ids): # don't re-download images
                
                file_name = media_id
                file_type = os.path.splitext(media_url)[1]

                urllib.request.urlretrieve(media_url, os.path.join(folder_name,file_name+file_type))
                
                downloaded_img_ids.append(media_id)
                
                # save image origin info
                image_origins["tweet_url"].append(tweet_url)
                image_origins["image_id"].append(media_id)
                image_origins["image_url"].append(media_url)




pickle.dump(image_origins,open("image_origins.p","wb"))
pickle.dump(all_results,open("all_results.p","wb"))




# turn dictionary to dataframe
image_info_df = pd.DataFrame.from_dict(image_origins)
# remove rows with duplicate image IDs
image_info_df = image_info_df.drop_duplicates(subset='image_id', keep='first')
# order columns
image_info_df = image_info_df[['image_id','image_url','tweet_url']]




#image_info_df.head()





#len(image_info_df)



image_info_df.to_csv("image_origins.csv",index=False)