from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import time
import warnings


# Suppress FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

# Set your Huggingface API token
huggingfacehub_api_token = "your_token_here"


# Initialize the client with a longer timeout (though this is more for `InferenceClient`)
client = InferenceClient(token=huggingfacehub_api_token, timeout=60)

# Define the LLM (HuggingFaceEndpoint from langchain_huggingface)
llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-Nemo-Instruct-2407',
    task="text-generation",
    huggingfacehub_api_token=huggingfacehub_api_token
)

# Predict the words following the input text
question = "Whatever you do, take care of your shoes."

# Debugging: Print the question and model ID
print(f"Question: {question}")
print(f"Model: {llm.repo_id}")

# Use the 'invoke' method to get a prediction (for langchain)
output = llm.invoke(question)

# Print the output
print("Model Output:", output)

