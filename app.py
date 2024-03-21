import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
import os
from chain import folder_selector, get_conversation_chain





os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Finpro_gemni"

LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "ls__f7a4bd725d8d4e709bd5a82a346623b6"
LANGCHAIN_PROJECT = "Finpro_gemni"

memory = ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
        return_messages=True,
        memory_key="chat_history",
    )
def main():
    st.set_page_config(page_title="Finpro.ai - FinGainInsights", page_icon=":moneybag:")
    st.title("Finpro - EarningsWhisperer 💹")
    client = Client(api_url=LANGCHAIN_ENDPOINT, api_key=os.environ['LANGCHAIN_API_KEY'])

    initialize_state()
    path = folder_selector()

   
    chain = get_conversation_chain(path, memory)

    handle_clear_message_history()
    

    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(
        callbacks=[run_collector],
        tags=["Streamlit Chat"],
    )
    display_trace_link()

    handle_user_input(chain, runnable_config, run_collector, client)
    display_chat_messages()
    handle_feedback(client)

def initialize_state():
    if "trace_link" not in st.session_state:
        st.session_state.trace_link = None
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "langchain_messages" not in st.session_state:
        st.session_state.langchain_messages =""

def handle_clear_message_history():
    if st.sidebar.button("Clear message history"):
        print("Clearing message history")
        memory.clear()
        st.session_state.trace_link = None
        st.session_state.run_id = None
        st.session_state.langchain_messages = []

def display_chat_messages():
    for msg in st.session_state.langchain_messages:
        avatar = "💰" if msg.type == "ai" else None
        with st.chat_message(msg.type, avatar=avatar):
            print(msg.content)
            st.markdown(msg.content)

def display_trace_link():
    if st.session_state.trace_link:
        st.sidebar.markdown(
            f'<a href="{st.session_state.trace_link}" target="_blank"><button>Latest Trace: 🛠️</button></a>',
            unsafe_allow_html=True,
        )

def reset_feedback():
    st.session_state.feedback_update = None
    st.session_state.feedback = None


def handle_user_input(chain, runnable_config, run_collector, client):
    if prompt := st.chat_input(placeholder="Ask a question about the Streamlit docs!"):
        st.chat_message("user").write(prompt)
        reset_feedback()
        with st.chat_message("assistant", avatar="💰"):
            message_placeholder = st.empty()
            full_response = ""
            input_structure = {"input": prompt}
            input_structure = {
                "question": prompt,
                "chat_history": [
                    (msg.type, msg.content)
                    for msg in st.session_state.langchain_messages],
            }

            for chunk in chain.stream(input_structure, config=runnable_config):
                full_response += chunk["answer"]
                print(full_response)
                message_placeholder.markdown(full_response + "")
                
            memory.save_context({"input": prompt}, {"output": full_response})
            message_placeholder.markdown(full_response)
            run = run_collector.traced_runs[0]
            run_collector.traced_runs = []
            st.session_state.run_id = run.id
            wait_for_all_tracers()
            url = client.share_run(run.id)
            st.session_state.trace_link = url

def handle_feedback(client):
    has_chat_messages = len(st.session_state.get("langchain_messages", [])) > 0

    if has_chat_messages:
        feedback_option = "faces"
    else:
        pass

    if st.session_state.get("run_id"):
        feedback = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{st.session_state.run_id}",
        )

        score_mappings = {
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        scores = score_mappings[feedback_option]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                feedback_record = client.create_feedback(
                    st.session_state.run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")

if __name__ == "__main__":
    main()