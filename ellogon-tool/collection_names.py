from api import Session, Collection, Document, Span, Attribute, EnhancedJSONEncoder
import json
from collections import Counter

session = Session(username="debatelab@ellogon.iit.demokritos.gr",
                  password="Up9F6AE2YN",
                  URL='https://vast.ellogon.org/');

##
## Get the list of user's collections...
##
collections = session.collections()
print(f"Number of Collections: {len(collections)}")
for col in collections:
    print(f"    Collection id: '{col.id}', name: '{col.name}'")

session.logout()
