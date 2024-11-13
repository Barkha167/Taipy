import os
from taipy.gui import Gui, State, notify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")



# Initial context and sample conversation
context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?"
conversation = {
    "Conversation": ["Who are you?", "Hi! I am GPT-4. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
rename_input = ""  # Input for renaming chat

# Initialize application state
def on_init(state: State) -> None:
    state.context = context
    state.conversation = conversation
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]
    # state.show_sidebar = True  # Sidebar is visible by default


# Generate response using the model
def request(state: State, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Update the conversation context with a new user message and AI response
def update_context(state: State) -> None:
    state.context += f"Human: \n {state.current_user_message}\n\n AI:"
    answer = request(state, state.context).replace("\n", "")
    state.context += answer
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer

# Send message and update conversation state
def send_message(state: State) -> None:
    notify(state, "info", "Sending message...")
    answer = update_context(state)
    conv = state.conversation.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.current_user_message = ""
    state.conversation = conv
    notify(state, "success", "Response received!")

# Rename the selected chat
def rename_chat(state: State) -> None:
    if state.selected_conv is not None:
        conv_id = state.selected_conv[0][0]
        state.past_conversations[conv_id][1]["Conversation"][0] = state.rename_input
        state.past_conversations[conv_id] = (conv_id, state.past_conversations[conv_id][1])  # Update the tuple with new name
        state.rename_input = ""
        notify(state, "success", "Chat renamed successfully!")
        # Refresh UI by updating the past conversations list
        state.past_conversations = state.past_conversations.copy()

# Delete the selected chat
def delete_chat(state: State) -> None:
    if state.selected_conv is not None:
        conv_id = state.selected_conv[0][0]
        state.past_conversations.pop(conv_id)
        state.selected_conv = None
        notify(state, "success", "Chat deleted successfully!")
        # Refresh UI by updating the past conversations list
        state.past_conversations = state.past_conversations.copy()

# Reset chat to start a new conversation
def reset_chat(state: State) -> None:
    # Save the current conversation to history before starting a new one
    if state.conversation["Conversation"]:
        state.past_conversations.append(
            [len(state.past_conversations), state.conversation]
        )
    # Start a fresh conversation
    state.conversation = {
        "Conversation": ["Who are you?", "Hi! I am GPT-4. How can I help you today?"]
    }
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today? "
    state.selected_conv = None  # Deselect any past conversation
    state.selected_row = [1]  # Reset the row selection

# Adapter for tree structure in UI
def tree_adapter(item: list) -> [str, str]:
    identifier = item[0]
    if len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][0][:50] + "...")
    return (item[0], "Empty conversation")

# Select a conversation from the history
def select_conv(state: State, var_name: str, value) -> None:
    if value:  # Ensure there is a selected conversation
        # Load the selected conversation
        conv_id = value[0][0]
        state.conversation = state.past_conversations[conv_id][1]
        
        # Rebuild context from selected conversation
        state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today? "
        for i in range(2, len(state.conversation["Conversation"]), 2):
            state.context += f"Human: \n {state.conversation['Conversation'][i]}\n\n AI:"
            state.context += state.conversation["Conversation"][i + 1]
        
        # Update selected conversation and row
        state.selected_conv = value
        state.selected_row = [len(state.conversation["Conversation"]) + 1]

# Define page layout and UI components
page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Conversation **Bot**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### History ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>

<|{rename_input}|input|label=Rename chat|on_change=rename_chat|placeholder="Enter new chat name"|>
<|Rename Chat|button|on_action=rename_chat|class_name=fullwidth mt-1|>
<|Delete Chat|button|on_action=delete_chat|class_name=fullwidth mt-1 danger|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    load_dotenv()
    Gui(page).run(debug=True, dark_mode=True, use_reloader=True, title="💬 Taipy Chat")
