import os
import pandas as pd
from IPython.display import Markdown, HTML, display
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from sqlalchemy import create_engine

class DBAgent:

    def __init__(self):
        self.client = AzureChatOpenAI(
            openai_api_version="2024-04-01-preview",
            azure_deployment="gpt-4-1106",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def human_message(self):
        message = HumanMessage(
            content="Translate this sentence from English "
                    "to French and Spanish. I like red cars and "
                    "blue houses, but my dog is yellow."
        )
        self.client.invoke([message])

    def talk_with_csv(self):
        df = pd.read_csv("./data/all-states-history.csv").fillna(value=0)
        agent = create_pandas_dataframe_agent(llm=self.client, df=df, verbose=True)

        agent.invoke("how many rows are there?")

        CSV_PROMPT_PREFIX = """
        First set the pandas display options to show all the columns,
        get the column names, then answer the question.
        """

        CSV_PROMPT_SUFFIX = """
        - **ALWAYS** before giving the Final Answer, try another method.
        Then reflect on the answers of the two methods you did and ask yourself
        if it answers correctly the original question.
        If you are not sure, try another method.
        - If the methods tried do not give the same result,reflect and
        try again until you have two methods that have the same result.
        - If you still cannot arrive to a consistent result, say that
        you are not sure of the answer.
        - If you are sure of the correct answer, create a beautiful
        and thorough response using Markdown.
        - **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
        ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
        - **ALWAYS**, as part of your "Final Answer", explain how you got
        to the answer on a section that starts with: "\n\nExplanation:\n".
        In the explanation, mention the column names that you used to get
        to the final answer.
        """

        QUESTION = "How may patients were hospitalized during July 2020"
        "in Texas, and nationwide as the total of all states?"
        "Use the hospitalizedIncrease column"

        agent.invoke(CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX)

    def talk_with_sql_db(self):
        # Path to your SQLite database file
        database_file_path = "./db/test.db"

        # Create an engine to connect to the SQLite database
        # SQLite only requires the path to the database file
        engine = create_engine(f'sqlite:///{database_file_path}')
        file_url = "./data/all-states-history.csv"
        df = pd.read_csv(file_url).fillna(value=0)
        df.to_sql(
            'all_states_history',
            con=engine,
            if_exists='replace',
            index=False
        )

        MSSQL_AGENT_PREFIX = """

        You are an agent designed to interact with a SQL database.
        ## Instructions:
        - Given an input question, create a syntactically correct {dialect} query
        to run, then look at the results of the query and return the answer.
        - Unless the user specifies a specific number of examples they wish to
        obtain, **ALWAYS** limit your query to at most {top_k} results.
        - You can order the results by a relevant column to return the most
        interesting examples in the database.
        - Never query for all the columns from a specific table, only ask for
        the relevant columns given the question.
        - You have access to tools for interacting with the database.
        - You MUST double check your query before executing it.If you get an error
        while executing a query,rewrite the query and try again.
        - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
        to the database.
        - DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
        OF THE CALCULATIONS YOU HAVE DONE.
        - Your response should be in Markdown. However, **when running  a SQL Query
        in "Action Input", do not include the markdown backticks**.
        Those are only for formatting the response, not for executing the command.
        - ALWAYS, as part of your final answer, explain how you got to the answer
        on a section that starts with: "Explanation:". Include the SQL query as
        part of the explanation section.
        - If the question does not seem related to the database, just return
        "I don\'t know" as the answer.
        - Only use the below tools. Only use the information returned by the
        below tools to construct your query and final answer.
        - Do not make up table names, only use the tables returned by any of the
        tools below.

        ## Tools:

        """

        MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

        ## Use the following format:

        Question: the input question you must answer.
        Thought: you should always think about what to do.
        Action: the action to take, should be one of [{tool_names}].
        Action Input: the input to the action.
        Observation: the result of the action.
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer.
        Final Answer: the final answer to the original input question.

        Example of Final Answer:
        <=== Beginning of example

        Action: query_sql_db
        Action Input: 
        SELECT TOP (10) [death]
        FROM covidtracking 
        WHERE state = 'TX' AND date LIKE '2020%'

        Observation:
        [(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
        Thought:I now know the final answer
        Final Answer: There were 27437 people who died of covid in Texas in 2020.

        Explanation:
        I queried the `covidtracking` table for the `death` column where the state
        is 'TX' and the date starts with '2020'. The query returned a list of tuples
        with the number of deaths for each day in 2020. To answer the question,
        I took the sum of all the deaths in the list, which is 27437.
        I used the following query

        ```sql
        SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
        ```
        ===> End of Example

        """



        db = SQLDatabase.from_uri(f'sqlite:///{database_file_path}')
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        QUESTION = """How may patients were hospitalized during October 2020
        in New York, and nationwide as the total of all states?
        Use the hospitalizedIncrease column
        """

        agent_executor_SQL = create_sql_agent(
            prefix=MSSQL_AGENT_PREFIX,
            format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
            llm=self.client,
            toolkit=toolkit,
            top_k=30,
            verbose=True
        )

        agent_executor_SQL.invoke(QUESTION)
