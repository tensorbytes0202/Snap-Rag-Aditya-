chat_history = []

def add_message(role,content):

    chat_history.append({
        "role": role,
        "content": content
    })

def get_history():

    return chat_history

def clear_history():

    global chat_history

    chat_history = []