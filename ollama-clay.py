"""
Example script for running Ollama with Argilla, a mixture of RAG and HF.
Argilla: https://github.com/argilla-io/argilla
Ollama: https://github.com/ollama/ollama-python (Requires client: ollama.com)

To run, fill in each <...> with values that match.
"""


import argilla as rg
from sentence_transformers import SentenceTransformer
from ollama import Client


client = Client(host='http://localhost:11434')
model = SentenceTransformer("thenlper/gte-small")  # gte-small used as example | replace with embedding model of choice
response_threshold = 5  # Put in the number of user responses needed to build confidence in adding the hf to vs
max_rating = 5  # Put the highest score from the dataset ratings
input_placeholder = "Placeholder to prompt user..."


# Argilla config:
rg.init(
    api_url="https://<username>-<spacename>.hf.space",
    api_key="owner.apikey",
    workspace="<workspacename>",
)

remote_dataset = rg.FeedbackDataset.from_argilla(
    name="<nameofdataset>",
    workspace="<workspacename>",
    with_vectors="all"
)


user_question = input(input_placeholder)
query_embedding = model.encode(user_question).tolist()
similar_records = remote_dataset.find_similar_records(vector_name="<nameofvectorsettings>", value=query_embedding)
hf_value = {index: [0, 0.0] for index, _ in enumerate(similar_records)}
vs_hf_list = []
for idx, record in enumerate(similar_records):
    rec, distance = record
    if rec.responses:
        submitted = [r for r in record.responses if r.status == "submitted"]
        if len(submitted) >= response_threshold:
            for response_ix, response in enumerate(submitted):
                hf_value[idx][0] += response.values["rating"].value / max_rating  # normalizing the value to distance
            hf_value[idx][1] = hf_value[idx][0] / len(submitted)
    final_score = hf_value[idx][1] + distance
    vs_hf_list.append([final_score, rec])
rag_from_vs_hs = sorted(vs_hf_list, key=lambda x: x[0], reverse=True)[0]


# Ollama config:
response = client.chat(model='<ollamamodel>', messages=[
  {
    'role': 'system',
    'content': 'Answer user questions using context they provide with their question.',
  },
  {
    'role': 'user',
    'content': "Context: " + rag_from_vs_hs[1].fields['answer'] + "\n" + "Question: " + user_question,
  },
])

print(response['message']['content'])
