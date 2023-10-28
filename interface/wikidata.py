# import sys
# from SPARQLWrapper import SPARQLWrapper, JSON

# query = """
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX wdt: <http://www.wikidata.org/prop/direct/>

# select distinct ?type ?typeLabel where {
#   ?thing wdt:P31 ?type
#   OPTIONAL {
#     ?type rdfs:label ?typeLabel filter (lang(?typeLabel) = "en").
#   }
# }
# limit 20
# """

# user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
# endpoint_url = "https://query.wikidata.org/sparql"
# sparql = SPARQLWrapper(endpoint_url, agent=user_agent)

# sparql.setQuery(query)
# sparql.setReturnFormat(JSON)

# result = sparql.queryAndConvert()


# def convert_response_for_data_frame(query_result):
#     columns = query_result["head"]["vars"]
#     result = []
#     for row in query_result["results"]["bindings"]:
#         column_values = []
#         for column in columns:
#             if column in row:
#                 column_values.append(row[column]["value"])
#             else:
#                 column_values.append(None)
#         result.append(column_values)

#     return (result, columns)


# print(convert_response_for_data_frame(result))

import json

with open("wikidata-20231025-lexemes.json") as user_file:
    parsed_json = json.load(user_file)

filtered = []
for item in parsed_json:
    if "en" in item["lemmas"]:
        filtered.append(item)
print(len(filtered))
print(filtered[0])
breakpoint()
