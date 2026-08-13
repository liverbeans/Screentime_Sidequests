import gradio as gr
from huggingface_hub import InferenceClient


# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

import torch

with open("research.txt", "r", encoding="utf-8") as file:
  # Read the entire contents of the file and store it in a variable
  research_text = file.read()
    
def preprocess_text(text):
  # Strip extra whitespace from the beginning and the end of the text
  cleaned_text = text.strip()

  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n")

  # Create an empty list to store cleaned chunks
  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  for chunk in chunks:
    stripped_chunk = chunk.strip()
    if len(stripped_chunk)>0:
      cleaned_chunks.append(stripped_chunk)

  # Print cleaned_chunks
  print(cleaned_chunks)

  # Print the length of cleaned_chunks
  print(len(cleaned_chunks))

  # Return the cleaned_chunks
  return cleaned_chunks

# Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(research_text) # Complete this line



def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)

  # Return the chunk_embeddings
  return chunk_embeddings

# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(cleaned_chunks) # Complete this line

# Define a function to find the most relevant text chunks for a given query, chunk_embeddings, and text_chunks
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
    chunk= text_chunks[i]
    top_chunks.append(chunk)
    
  return top_chunks
 
  # Return the list of most relevant chunks
def respond(name, message, history, intrests, ages, time):
    top_results = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    context = "\n".join(top_results)
    
    activities_str = ", ".join(activities) if activities else "no specific activities"
    
    system_prompt = (
    f"Always refer to the user by {name}."
    f"You are a chill chatbot who is encouraging and uses emojis, and really tries to get people off their devices. "
    f"Use the following research context to help answer questions:\n\n{context}\n\n"
    f"IMPORTANT: The user has {time} this amount of time. Tailor your suggestions specifically for a {time} as well as {ages} and {intrests_str}"
    f"IMPORTANT: The user is a {ages}. Tailor your suggestions specifically for a {ages}"
    f"— use age-appropriate language and activity difficulty. "
    f"They're especially interested in: {intrests_str}.")
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    response = client.chat_completion(messages, max_tokens=2000, temperature=0.5)
    return response.choices[0].message.content.strip()   # <- stays last, unindented from the block above it


chatbot = gr.ChatInterface(respond,
   
     title="｡  🎀  𝒮𝒸𝓇𝑒𝑒𝓃𝓉𝒾𝓂𝑒 𝒮𝒾𝒹𝑒𝓆𝓊𝑒𝓈𝓉𝓈  🎀  ｡",
                           
    description = "Input your age, interests, and time you want to spend off your screen! You can do this in the text box if you want to be more specific with things like your budget, or you can click the checkboxes by opening up the additional inputs! ", 
    
    additional_inputs=[ gr.Textbox(label="Your Name"), gr.CheckboxGroup (["shopping", "art","sports/working out", "cooking"," reading","video games", "music","writing"],label="intrests"), 
    gr.Radio (["thirty minutes","one hour","two hours","three hours","four hours","five+ hours"],label="time"),
     
    gr.Radio (["5-7", "8-10", "11-13","14-17","18-22","23-25","25-30","31-40","41-50","50+"], label="ages") ] )

chatbot.launch(theme="NeoPy/BoyKisser")




# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
