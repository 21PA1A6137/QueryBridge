
# def generate_response(user_input, engine):
#     if engine is None:
#         return "Please upload your data first."

#     try:
#         db = SQLDatabase(engine)
#         llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0)

#         agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
#         response = agent_executor.invoke({"input": user_input})

#         sql_query = response.get("sql_query", None)
#         if sql_query:
#             st.write(f"Generated SQL Query: {sql_query}")

#             # Execute the query
#             with engine.connect() as conn:
#                 result = conn.execute(text(sql_query))
#                 rows = result.fetchall()

#                 if rows:
#                     return "\n".join([str(row) for row in rows])
#                 else:
#                     return "No data found."
#         else:
#             return "No query generated."

#     except Exception as e:
#         return f"Error: {e}"