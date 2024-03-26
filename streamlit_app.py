#Write a simple app that reads the user input and display the output
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

document_text = "SpaceX: Aiming for the Stars SpaceX is driven by an ambitious mission and vision\
for the future of space exploration. Here's a breakdown of their key goals and objectives: \
Mission:  Revolutionize space technology, with the ultimate goal of enabling people to live \
on other planets. This two-pronged approach emphasizes innovation in space travel technology \
while keeping a long-term vision of human Martian settlements.  Their focus on reusable rockets \
and cost reduction exemplifies their mission to make space travel more accessible. \
Vision (unofficial):** Make life multi-planetary by establishing a self-sustaining city \
on Mars. While SpaceX doesn't have an official vision statement, their actions speak volumes.  \
The development of the Starship program, a fully reusable launch vehicle and spacecraft, is a clear \
indicator of their vision for interplanetary travel and colonization.  \
**Goals and Objectives:**\
* Develop and implement next-generation, fully reusable launch vehicles (Starship and \
Super Heavy) for cost-effective space travel. [2] * Reduce the cost of space access \
through innovation and reusability. \
* Secure future resources and ensure the long-term survival of humanity through a multi-planetary future.  \
* Foster international collaboration in space exploration endeavors. \
SpaceX's mission, vision, and goals all work together to push the boundaries of space exploration. \
They are constantly striving to make space travel more affordable and reliable, paving the way \
for a future where humanity is no longer confined to Earth."

def process_document(document):
    """
    Preprocesses the document for model input.

    Args:
        document: String containing the document text.

    Returns:
        Processed document as a string.
    """
    # You can add pre-processing steps here like:
    # - Removing stop words
    # - Summarizing the document
    # - Converting to a specific format
    return document

# Define the Streamlit app
def app():
    # Access the environment variable set by Streamlit Secrets Management
    access_token = os.environ.get("HUGGINGFACE_API_KEY")

    # Model selection (choose between base or instruction-tuned variant)
    model_name = "google/gemma-2b-it"  # Example: Instruction-tuned Gemma 2B

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

    st.header("Welcome to Google Gemma AI")
    st.subheader("Louie F. Cervantes M. Eng. \n(c) 2024 WVSU College of ICT")

    # Process the document
    processed_document = process_document(document_text)

    if st.button("Submit"):
        # Get user input
        user_input = st.text_input("You:")

        # Preprocess conversation history (you can modify this)
        history = f"Document: {processed_document}\nYou: {user_input}"

        # Encode the conversation history
        input_ids = tokenizer(history, return_tensors="pt")["input_ids"]

        # Generate response from the model
        output = model.generate(input_ids, max_length=1000, do_sample=True)

        # Decode the generated response
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Print the model response
        st.write(f"Gemma: {response}")


# Run the app
if __name__ == "__main__":
    app()
