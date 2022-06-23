# -*- coding: utf-8 -*-
from ellogon import sentiment
from ellogon import expandnames
from elasticsearch     import Elasticsearch
from elasticsearch_dsl import Search, Q
from elasticsearch.helpers import bulk

def analyse_articles(EID, period, client, name=None):
    index  = 'articles'
    indexs = 'scify-articles-slice'
    query = expandnames.get_es_query(EID, fields=['content','title','tags'], client=client)
    client.indices.refresh(index=index)
    s = Search(using=client, index=index) \
            .exclude("range", date={'lte': period}) \
            .query("bool", filter=query)\
            .sort("date", "desc")
    count = 0
    docs  = []
    slice_articles = []
    # Use a small size so as searh context does not timeout (5minutes)
    for hit in s.params(size=100).scan():
        try:
            print(name, '(', count, 'a)')
            doc = sentiment.process_article(hit, client=None)
            count += 1
            if doc:
                docs.append(doc)
                slice_articles.append({'_index': indexs,
                    '_type': hit.meta.doc_type,
                    '_id': hit.meta.id,
                    '_source': hit._d_}
                )
                # client.index(index=indexs, doc_type=hit.meta.doc_type,
                #     id=hit.meta.id, body=hit._d_)
            else:
                print(hit.title + " . \n\n" + hit.content)
        except Exception as e:
            print("Error:", hit.meta.id)
            print(hit.text)
            print(e)
    if len(docs):
        print("Bulk:", bulk(client, docs))
        print("Bulk slice:", bulk(client, slice_articles))
        client.indices.refresh(index='sentiment')
        client.indices.refresh(index=indexs)
    return count

def analyse_tweets(EID, period, client, name=None):
    index  = 'tweets'
    query = expandnames.get_es_query(EID, fields=['text'], client=client)
    client.indices.refresh(index=index)
    s = Search(using=client, index=index) \
            .exclude("range", date={'lte': period}) \
            .exclude("exists", field='retweeted_status') \
            .query("bool", filter=query)\
            .sort("date", "desc")
    count = 0
    docs  = []
    for hit in s.params(size=100).scan():
        try:
            print(name, '(', count, 't)')
            doc = sentiment.process_tweet(hit, client=None)
            count += 1
            if doc:
                docs.append(doc)
            else:
                print(hit.text)
        except Exception as e:
            print("Error:", hit.meta.id)
            print(hit.text)
            print(e)
    if len(docs):
        print("Bulk:", bulk(client, docs))
        client.indices.refresh(index='sentiment')
    return count

def analyse_comments(EID, period, client, name=None):
    index  = 'comments'
    query = expandnames.get_es_query(EID, fields=['message'], client=client)
    client.indices.refresh(index=index)
    s = Search(using=client, index=index) \
            .exclude("range", date={'lte': period}) \
            .query("bool", filter=query)\
            .sort("date", "desc")
    count = 0
    docs  = []
    for hit in s.params(size=100).scan():
        try:
            print(name, '(', count, 'c)')
            doc = sentiment.process_comment(hit, client=None)
            count += 1
            if doc:
                docs.append(doc)
            else:
                print(hit.message)
        except Exception as e:
            print("Error:", hit.meta.id)
            print(hit.text)
            print(e)
    if len(docs):
        print("Bulk:", bulk(client, docs))
        client.indices.refresh(index='sentiment')
    return count
