import streamlit as st
import os
import pandas as pd
from pyprojroot import here
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine, inspect,text
from dotenv import load_dotenv



st.markdown(
    """
    <style>
    .fixed-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True
)







# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "engine" not in st.session_state:
    st.session_state["engine"] = None

# Initialize the SQL engine using SQLite
def initialize_sql_engine():
    db_file = os.path.join(here("configs"), "upload_files_sqldb_dir", "database.db")
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_file}")
    st.session_state["engine"] = engine
    return engine

# Save uploaded files to the SQL database
def save_uploaded_files_to_sql(uploaded_files, engine):
    table_names = []
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return

        if df.empty:
            st.error(f"The file '{file.name}' is empty. Please upload a non-empty CSV or XLSX file.")
            continue

        table_name = file.name.split(".")[0]
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        # st.write(f"Saved {file.name} to SQL table '{table_name}'")
        table_names.append(table_name)
        st.write(df.head(5))

    return table_names  # Return the list of table names created

# List available tables in the database
def list_tables(engine):
    if engine is None:
        st.write("No engine found. Please upload a file first.")
        return []

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    # st.write("Available tables:", tables)
    return tables

def generate_response(user_input, engine):
    if engine is None:
        return "Please upload your data first."

    try:
        # Retrieve table schema information
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return "No tables found in the database. Please upload your data."

        # Generate table schema info to use in prompt
        schema_info = ""
        for table in tables:
            columns = inspector.get_columns(table)
            schema_info += f"Table: {table}\nColumns: {', '.join([col['name'] for col in columns])}\n"

        # Initialize database and Groq model
        db = SQLDatabase(engine)
        llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0)

        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

        # Prepare the prompt
        agent_prompt = f"""You are an AI agent that generates SQL queries for a relational database. 
        The database contains the following tables and columns:\n{schema_info}\n
        Based on the user's input, generate a SQL query and return the result.

        User question: {user_input}
        Please provide the SQL query to execute.
        """

        # Get response from agent
        response = agent_executor.invoke({"input": agent_prompt})

        # Debug the response to see its structure
        # st.write("Raw response from agent:", response)

        # Extract the result from the response
        if response and "output" in response:
            result_output = response["output"]
            # st.write(f"Generated result: {result_output}")
            return result_output
        else:
            return "No query generated or no result found."

    except Exception as e:
        return f"Error generating SQL query: {e}"




# Streamlit main function
def main():
    st.set_page_config(page_title="QueryBridge - Talk to your data!", layout="wide")
    
    st.title("QueryBridge - Talk to your data!")

    tab1, tab2 = st.tabs(["Talk to your CSV/XLSX data", "Chat with your PDFs"])

    with tab1:
        st.subheader("Chat Interface")

        # Initialize the SQL engine if it's not already initialized
        if st.session_state["engine"] is None:
            st.session_state["engine"] = initialize_sql_engine()

        # Sidebar for file uploads
        uploaded_files = st.sidebar.file_uploader("Upload Files", type=["csv", "xlsx"], accept_multiple_files=True)

        if uploaded_files:
            st.write("Processing uploaded files...")
            save_uploaded_files_to_sql(uploaded_files, st.session_state["engine"])

        # Display available tables in the database
        list_tables(st.session_state["engine"])

        # Chat interface for asking questions
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # prompt = st.chat_input("What is your question?")
        # # Handle user input and generate SQL queries based on the input
        # if prompt:
        #     st.session_state["messages"].append({"role": "user", "content": prompt})
        #     with st.chat_message("user"):
        #         st.markdown(prompt)

        #     with st.chat_message("assistant"):
        #         response = generate_response(prompt, st.session_state["engine"])
        #         st.markdown(response)
        #     st.session_state["messages"].append({"role": "assistant", "content": response})

        with st.container():
            user_input = st.text_input("What is your question?", key="user_input", label_visibility="collapsed")

            if user_input:
                st.session_state["messages"].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    response = generate_response(user_input, st.session_state["engine"])
                    st.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})


        # Clear chat button
        if st.sidebar.button("Clear Chat"):
            st.session_state["messages"] = []

    with tab2:
        st.subheader("PDF Interaction (Coming Soon)")
        st.write("This feature is under development and will support chatting with PDFs.")

if __name__ == "__main__":
    main()
