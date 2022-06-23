import sys
import requests
import logging
import json
from src.utils import read_json_str_or_file


class CrawlerProber:
    """Middleware between crawler, ES and the rest API caller"""

    def __init__(self, crawler_creds_file, es_retrieve=None):
        """
        Constructor
        Args:
            crawler_creds_file: Crawler credentials
            es_retrieve: ElasticSearch store configuration

        Returns:

        """
        self.es_retrieve = es_retrieve
        with open(crawler_creds_file) as f:
            self.credentials = tuple([x.strip() for x in f.readlines()])

    def probe(self, url, retrieve_result=True):
        """
        Probe the crawler with a url.
        Args:
            url: article URL to crawl

        Returns:
            The crawled document contents
        """
        payload = {"request": {
            "url": url
        },
            "spider_name": "genericurl",
            "crawl_args": {}
        }
        auth = self.credentials
        args = {"url": "https://socialwebobservatory.iit.demokritos.gr/crawl.json", "json": payload}
        if auth:
            args["auth"] = auth
        response = requests.post(**args)
        response = json.loads(response.content)
        if not retrieve_result:
            return response
        lookup_link = url
        try:
            article_not_already_in_es_store = bool(response["items"])
            if article_not_already_in_es_store:
                doc = response["items"][0]
                logging.info(f"Crawled article {url} from scratch.")
                # new article was crawled -- make sure you use the stored link
                # in case any redirect occurred
                lookup_link = doc["link"]
                # if we throw an exception here, next call with the same url retrieves the article from ES
                # with the title inserted as the content
                # if doc['content'] == '':
                #     raise ValueError("No article content could be parsed.")
            else:
                logging.info(f"Discovered article {url} already in SWO ES.")

            # by now article exists in the ES store; fetch it
            args = {"mode": "data", "data": {"type": "link", "values": [lookup_link]}}
            docs = self.es_retrieve.retrieve_documents(**args)
            if not docs:
                raise ValueError(f"No article(s) retrieved for url: {url}.")
            doc = docs[0]
            if doc['content'] == '':
                raise ValueError("Retrieved stored article with empty content.")
            return doc, ""
        except Exception as exc:
            logging.error(f"Unable to crawl content from url [{url}]")
            logging.error(f"{exc}")
            return None, f"Unable to crawl doc: [{url}]"


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        links = [x for x in [x.strip() for x in f.readlines()] if x]
    crawler = CrawlerProber("../crawler_credentials.txt")
    issues = {}
    for i, url in enumerate(links):
        print("Crawling link", i + 1, "/", len(links), ":", url)
        response = crawler.probe(url, retrieve_result=False)
        status = response['status']
        if status != 'ok':
            issues[url] = response
    print(issues)
