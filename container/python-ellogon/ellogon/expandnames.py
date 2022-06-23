# -*- coding: utf-8 -*-
from ellogon import tcl, package_require
from elasticsearch     import Elasticsearch
from elasticsearch_dsl import Search, Q

package_require("morphlex::greek")
tcl.call("proc", "::GenerativeLexicon::log", "", "")

def name_synonyms(name, client=None, expand=True):
    index = "namesgreek"
    client.indices.refresh(index=index)
    s = Search(using=client, index=index).query("match", name=name)
    response = s.scan()
    synonyms = [str(name)]
    for hit in response:
        if "synonyms" in hit and hit.synonyms:
            if expand:
                for synonym in hit.synonyms:
                    synonyms.extend(name_synonyms(str(synonym).strip(), client, False))
            else:
                synonyms.extend([str(s).strip() for s in hit.synonyms])
    return [str(x) for x in synonyms]


def expand(id, first, surname, middle=None):
    cmd = ['::morphlex::greek::lexicon', 'personNameCombinations',
            first, surname]
    if middle:
        cmd += [middle]

    # data = tcl.call('puts', ' '.join(cmd))
    data = tcl.call(*cmd)
    #print(cmd)
    #print(data)
    return [str(x) for x in data]
    # print(data)
    # for one in data:
    #     print(one, id)
    # tcl2py = nestedExpr('{', '}')
    # #tcl2py = originalTextFor(nestedExpr('{', '}'))
    # py = tcl2py.searchString(data)
    # return [one[0] for one in py]
    # for one in py:
    #     print(one[0], id)

def process_person(hit, client):
    alts = []
    if 'first_name' in hit and 'last_name' in hit:
        first   = hit.first_name.strip()
        surname = hit.last_name.strip()
        middle  = None
        if 'fathers_name' in hit:
            middle  = hit.fathers_name.strip()
        for fn in name_synonyms(first, client):
            result = expand(hit.meta.id, fn, surname, middle)
            if result: alts.extend(result)
            if middle:
                result = expand(hit.meta.id, fn, surname)
                if result: alts.extend(result)
    if 'name' in hit:
        alts.extend([hit.name.strip()])
        items   = [s.strip() for s in hit.name.split()]
        first   = items[0]
        middle  = None
        if len(items) == 2:
            surname = items[1]
        elif len(items) == 3:
            middle  = items[1]
            surname = items[2]
        else:
            raise ValueError("invalid name: " + hit.name)
        for fn in name_synonyms(first, client):
            result = expand(hit.meta.id, fn, surname, middle)
            if result: alts.extend(result)
    return alts

def process_party(hit, client):
    pass

def process_organisation(hit, client):
    pass

def process_location(hit, client):
    pass

def process_other(hit, client):
    pass

def process_entity(hit, client=None):
    # print(hit.to_dict())
    alts = []
    negs = []
    if 'name' in hit:
        alts.extend([hit.name.strip()])
    if 'category' in hit:
        if hit.category   == "Person":       result = process_person(hit, client)
        elif hit.category == "Party":        result = process_party(hit, client)
        elif hit.category == "Organization": result = process_organisation(hit, client)
        elif hit.category == "Location":     result = process_location(hit, client)
        elif hit.category == "Other":        result = process_other(hit, client)
        else: raise ValueError("invalid category: " + hit.category)
        if result: alts.extend(result)
    if 'keywords'         in hit and hit.keywords: alts.extend([s.strip() for s in hit.keywords])
    if 'hashtags'         in hit and hit.hashtags:
        alts.extend([s.strip() for s in hit.hashtags])
        alts.extend([s.strip(' #') for s in hit.hashtags])
    if 'mentions'         in hit and hit.mentions: alts.extend([s.strip() for s in hit.mentions])
    if 'twitter_username' in hit and hit.twitter_username: alts.extend([hit.twitter_username.strip()])
    ## Negative patterns...
    if 'excluded_keywords' in hit and hit.excluded_keywords: negs.extend([s.strip() for s in hit.excluded_keywords])
    return sorted(set(alts)), sorted(set(negs))

def get_alternatives_for_entity(EID, client=None):
    index    = 'app_entities'
    client.indices.refresh(index=index)
    s = Search(using=client, index=index).query("term", _id=EID)
    response = s.scan()
    alts = []
    negs = []
    for hit in response:
        a, n = process_entity(hit, client)
        alts.extend(a)
        negs.extend(n)
    return alts, negs

def get_es_query(EID, fields=['content','title','tags'], n=1000, client=None):
    alts, negs = get_alternatives_for_entity(EID, client)
    chunks = [alts[i:i + n] for i in range(0, len(alts), n)]
    q = '"' + '" "'.join(chunks[0]) + '"'
    query = Q("query_string", query=q, default_operator="OR",
               fields=fields)
    for chunk in chunks[1:]:
        q = '"' + '" "'.join(chunk) + '"'
        query |= Q("query_string", query=q, default_operator="OR",
               fields=fields)
    ## Negative patterns...
    chunks = [negs[i:i + n] for i in range(0, len(negs), n)]
    for chunk in chunks:
        q = '"' + '" "'.join(chunk) + '"'
        query &= ~Q("query_string", query=q, default_operator="OR",
               fields=fields)
    return query
