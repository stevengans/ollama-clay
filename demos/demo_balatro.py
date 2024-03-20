import argilla as rg
from sentence_transformers import SentenceTransformer
from ollama import Client


client = Client(host='http://localhost:11434')
model = SentenceTransformer("thenlper/gte-small")
response_threshold = 5
max_rating = 5
input_placeholder = "What would you like to know about Balatro?"


rg.init(
    api_url="https://sgans-balatro.hf.space",
    api_key="owner.apikey",
    workspace="guide",
)

remote_dataset = rg.FeedbackDataset.from_argilla(
    name="instruct",
    workspace="guide",
    with_vectors="all"
)


user_question = input(input_placeholder)
query_embedding = model.encode(user_question).tolist()
similar_records = remote_dataset.find_similar_records(vector_name="sentence_embeddings", value=query_embedding)
hf_value = {index: [0, 0.0] for index, _ in enumerate(similar_records)}
vs_hf_list = []
for idx, record in enumerate(similar_records):
    rec, distance = record
    if rec.responses:
        submitted = [r for r in record.responses if r.status == "submitted"]
        if len(submitted) >= response_threshold:
            for response_ix, response in enumerate(submitted):
                hf_value[idx][0] += response.values["rating"].value / max_rating
            hf_value[idx][1] = hf_value[idx][0] / len(submitted)
    final_score = hf_value[idx][1] + distance
    vs_hf_list.append([final_score, rec])
rag_from_vs_hs = sorted(vs_hf_list, key=lambda x: x[0], reverse=True)[0]


response = client.chat(model='gemma:2b', messages=[
  {
    'role': 'system',
    'content': 'You are a joker playing card from the game Balatro. Answer user questions about the game using'
               'context they provide with their question. Add a chaotic but friendly tone to responses',
  },
  {
    'role': 'user',
    'content': "Context: " + rag_from_vs_hs[1].fields['answer'] + "\n" + "Question: " + user_question,
  },
])

print(response['message']['content'])
