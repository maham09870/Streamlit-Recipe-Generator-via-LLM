import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
st.title('Recipe Generator Application')
st.write("This application generates recipes based on your input ingredients.")
st.image("https://media.qrtiger.com/blog/2023/01/1-food-recipe-header-copyjpg_800.webp")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

PROMPT = ""

# client = genai.client(api_key = GOOGLE_API_KEY)

user_input = st.text_input("Enter ingredients (comma-separated)")
send_button = st.button("Generate Recipe")

if send_button:
    if user_input:
        PROMPT= f"Generate a recipe using the following ingredients: {user_input}"
        response = model.generate_content(PROMPT)
        if response and hasattr(response, "text"):
            st.markdown("### Generated Recipe")
            st.markdown(response.text)    
        else:
            st.warning("No recipe could be generated. Please try again with different ingredients.")
    