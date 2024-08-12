from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

import os
import streamlit as st
import re

st.set_page_config(page_title="Nurse: Chat with search", page_icon="😎")
st.title("😎 Nurse: Chat with search")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
os.environ["TAVILY_API_KEY"] = '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'

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("We are waiting...")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    if msg.type == "human":
        continue

    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)


def all_the_work(llm, prompt) -> str:

    tools = [TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
    )]  

    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        verbose=True,
        memory=memory,
    )


    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = executor.invoke(prompt, cfg)

        st.write(response['output'])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]


    return response['output']

def run(llm1, llm2, info, rating1=0, rating2=0):
    prompt_base = '''
                Your response should include:

                1. A fact or piece of information found through a Google search.
                2. One pro argument supporting the idea that English is the best choice as a universal language, limited to no more than three sentences.
                3. One counterargument against the idea that English should be the universal language, limited to no more than two sentences.
                4. One question to continue the conversation, related to the topic.

                Ensure that all four points are included in the response. Especially, information from the search should be included.
                Your opponent said:
              ''' 
    
    response2 = info


    for i in range(5):

        prompt_head = 'You strongly believe that English is the best language as future universal language for the world.'

        prompt1 = prompt_head + prompt_base + response2


        response1 = all_the_work(llm1, prompt1)


        prompt_head = 'You strongly believes that English is not the best language and that actually the best language is Chinese.'
        
        prompt2 = prompt_head + prompt_base + response1

        response2 = all_the_work(llm2, prompt2)

    st.write(''' Rate first and second bot using scale from 1 to 10. \nENGLISH: your rating \nCHINESE: your rating ''')

    return response2

if 'counter_of_answers' not in st.session_state:
    st.session_state.counter_of_answers = 0

if 'keep_rating' not in st.session_state:
    st.session_state.keep_rating = [0, 0]

if 'prompt_new' not in st.session_state:
    st.session_state.prompt_new = ''

if prompt := st.chat_input(placeholder="Start argue."):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if st.session_state.counter_of_answers == 0:
        llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        
        st.session_state.counter_of_answers += 1

        st.session_state.prompt_new = run(llm1, llm2, prompt)

    else:

        num = re.findall(r'\d+', prompt)

        rating1, rating2 = int(num[0]), int(num[1])

        llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True, temperature=rating1/10)
        llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True, temperature=rating2/10)      

        st.session_state.keep_rating[0] += rating1
        st.session_state.keep_rating[1] += rating2

        st.session_state.counter_of_answers += 1

        st.session_state.prompt_new = run(llm1, llm2, st.session_state.prompt_new, rating1, rating2)


    if st.session_state.counter_of_answers >= 4:
        st.write('Final scores:', st.session_state.keep_rating[0], 'for english and', st.session_state.keep_rating[1], 'for chinese')