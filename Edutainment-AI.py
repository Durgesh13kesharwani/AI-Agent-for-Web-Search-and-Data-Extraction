import os
import pandas as pd
import streamlit as st
import asyncio
import concurrent.futures
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, ChatSession
import gspread

# Set environment variable for Gemini API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\HP\Downloads\ace-forest-442118-g4-d61d09bcf366.json"  # Path to Gemini API credentials

# Initialize Vertex AI (Gemini API)
PROJECT_ID = "your-project-id"  # Replace with your Google Cloud Project ID
aiplatform.init(project=PROJECT_ID, location="us-central1")

# Set up Gemini model
model = GenerativeModel("gemini-1.5-flash-002")  # Replace with your Gemini model ID
chat_session = model.start_chat()

# Function to send a prompt and receive a response asynchronously
async def get_chat_response_async(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

# Streamlit UI for file upload and data handling
st.title("AI Agent for Web Search and Data Extraction")

# Step 1: Upload CSV or Google Sheet
st.header("Step 1: Upload Your Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
google_sheet_url = st.text_input("Enter your Google Sheet URL (optional):")

# Initialize 'data' as None
data = None

# If file is uploaded
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# If Google Sheet URL is provided
if google_sheet_url:
    try:
        # Authenticate and access the Google Sheet
        gc = gspread.service_account(filename=r"C:\Users\HP\Downloads\ace-forest-442118-g4-d61d09bcf366.json")  # Path to Google service account credentials
        spreadsheet = gc.open_by_url(google_sheet_url)
        sheet = spreadsheet.sheet1  # Read the first sheet
        data = pd.DataFrame(sheet.get_all_records())
        st.write("Preview of Google Sheet Data:")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"Error reading the Google Sheet: {e}")

# Step 2: Define the query
if data is not None:
    st.header("Step 2: Define Your Query")
    column = st.selectbox("Select the column for entities:", data.columns)
    prompt_template = st.text_input("Enter your query template (use {entity}):", "How did the S&P 500 index perform around {entity}?")
    entities = data[column].dropna().unique().tolist()
    st.write(f"Selected Column: {column}")
    st.write(f"Number of Entities: {len(entities)}")

    # Step 3: Generate responses using Gemini API asynchronously
    async def process_entities(entities, prompt_template, data, column):
        extracted_information = {}
        loop = asyncio.get_event_loop()
        # Create a thread pool executor to run the async function in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = []
            for entity in entities:
                # Formulate the search query for each entity
                search_query = prompt_template.replace("{entity}", str(entity))
                
                # Perform entity lookup in data to provide additional context (e.g., for a date)
                if column == "Date" and column in data.columns:  # Ensure 'Date' column exists in data
                    try:
                        # Assuming 'Date' is the column in the uploaded data
                        result = data[data[column] == entity]  # Find rows matching the entity
                        if not result.empty:
                            # Extract a specific value or column from the row (e.g., 'performance')
                            entity_value = result.iloc[0]['Performance']  # Adjust 'Performance' column as needed
                            search_query += f" The value on this date was {entity_value}."
                    except Exception as e:
                        search_query += f" Error retrieving entity value: {e}"

                # Run the async function for this search query
                task = loop.run_in_executor(executor, get_chat_response_async, chat_session, search_query)
                tasks.append(task)
            
            # Wait for all tasks to finish
            results = await asyncio.gather(*tasks)
            
            # Store the results in the dictionary
            for idx, entity in enumerate(entities):
                extracted_information[entity] = results[idx]
        return extracted_information

    if st.button("Search Entities"):
        st.write("Searching for relevant information...")

        # Run the entity processing asynchronously
        extracted_information = asyncio.run(process_entities(entities, prompt_template, data, column))

        st.success("Search and processing completed!")

        # Display extracted information
        st.header("Extracted Information")
        st.json(extracted_information)

        # Optionally, save the results to a CSV file
        def save_to_csv(data_dict):
            try:
                df = pd.DataFrame(list(data_dict.items()), columns=["Entity", "Extracted Info"])
                csv_path = "extracted_data.csv"
                df.to_csv(csv_path, index=False)
                return csv_path
            except Exception as e:
                st.error(f"Error saving data to CSV: {e}")
                return None

        # Provide download option for results
        if extracted_information:
            st.header("Download Extracted Results")
            extracted_df = pd.DataFrame(list(extracted_information.items()), columns=["Entity", "Extracted Info"])
            st.dataframe(extracted_df)
            csv_path = save_to_csv(extracted_information)
            if csv_path:
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="Download Results as CSV",
                        data=f.read(),
                        file_name="extracted_data.csv",
                        mime="text/csv"
                    )
else:
    st.error("Please upload a file or provide a Google Sheet URL.")
