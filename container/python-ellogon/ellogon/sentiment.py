# -*- coding: utf-8 -*-
import json
from ellogon import tcl, package_require, text2tcl
tcl.call("tcl::tm::path", "add", "/opt/Intellitech/projects/tm")
tcl.call("tcl::tm::path", "add", "/opt/Intellitech/projects/SocialWebObservatory/tm")

package_require("sentiment")
## Create the processor
tcl.call("::SocialWebObservatory::sentiment", "create", "::sentiment")
tcl.eval("puts \"Using Tcl version: \\\"[info patchlevel]\\\"\"")

def process_article(hit, client=None, type="article", dsource=1, index='sentiment', doc_type='_doc'):
    data = text2tcl(hit.title + " . \n\n" + hit.content)
    # print(data)
    result = tcl.call("::sentiment", "process", data,
                      ['id', hit.meta.id, 'webrunner_dsource', dsource])
    j = dict()
    j['id']             = hit.meta.id
    j['type']           = type
    j['date']           = hit.date
    j['title']          = hit.title
    j['link']           = hit.link
    j['netloc']         = hit.netloc
    j['domain']         = hit.domain
    j['polarityOveral'] = 0
    j.update(json.loads(str(result)))
    j['polarity'] = int(j['polarity'])
    if len(j['hasEntities']):
        if j['polarity'] > 0:
            j['polarityOveral'] = 1
        elif j['polarity'] < 0:
            j['polarityOveral'] = -1
        print(j)
        if client:
            if not doc_type: doc_type=hit.meta.type
            client.index(index=index, doc_type=doc_type, id='a:'+hit.meta.id, body=j)
            client.indices.refresh(index)
        return {'_index': index, '_type': doc_type, '_id': 'a:'+hit.meta.id, '_source': j}
    return None

def process_tweet(hit, client=None, type="tweet", dsource=5, index='sentiment', doc_type='_doc'):
    data = text2tcl(hit.text)
    # print(data)
    result = tcl.call("::sentiment", "process", data,
                      ['id', hit.meta.id, 'webrunner_dsource', dsource])
    j = dict()
    j['id']             = hit.meta.id
    j['type']           = type
    j['date']           = hit.date
    j['title']          = hit.text
    if 'retweeted_status' in hit:
        j['retweeted_status'] = hit.retweeted_status
    j['polarityOveral'] = 0
    j.update(json.loads(str(result)))
    j['polarity'] = int(j['polarity'])
    if len(j['hasEntities']):
        if j['polarity'] > 0:
            j['polarityOveral'] = 1
        elif j['polarity'] < 0:
            j['polarityOveral'] = -1
        print(j)
        if client:
            if not doc_type: doc_type=hit.meta.type
            client.index(index=index, doc_type=doc_type, id='t:'+hit.meta.id, body=j)
            client.indices.refresh(index)
        return {'_index': index, '_type': doc_type, '_id': 't:'+hit.meta.id, '_source': j}
    return None

def process_comment(hit, client=None, type="comment", dsource=1, index='sentiment', doc_type='_doc'):
    data = text2tcl(hit.message)
    # print(data)
    result = tcl.call("::sentiment", "process", data,
                      ['id', hit.meta.id, 'webrunner_dsource', dsource])
    j = dict()
    j['id']             = hit.meta.id
    j['type']           = type
    j['date']           = hit.date
    j['title']          = hit.message
    j['link']           = hit.articlelink
    j['netloc']         = hit.netloc
    j['domain']         = hit.domain
    j['thread']         = hit.thread
    j['polarityOveral'] = 0
    j.update(json.loads(str(result)))
    j['polarity'] = int(j['polarity'])
    if len(j['hasEntities']):
        if j['polarity'] > 0:
            j['polarityOveral'] = 1
        elif j['polarity'] < 0:
            j['polarityOveral'] = -1
        print(j)
        if client:
            if not doc_type: doc_type=hit.meta.type
            client.index(index=index, doc_type=doc_type, id='c:'+hit.meta.id, body=j)
            client.indices.refresh(index)
        return {'_index': index, '_type': doc_type, '_id': 'c:'+hit.meta.id, '_source': j}
    return None
