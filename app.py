import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import random


data = pd.read_csv('chatbot_dataset.csv', on_bad_lines='war' \
'n', engine='python')

# Preprocessing  data
nltk.download('punkt')
nltk.download('punkt_tab')  # needed by newer nltk versions
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Spliting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['Question'], data['Answer'], test_size=0.2, random_state=42
)

# Creating a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Model training
model.fit(X_train, y_train)

# Evaluate model 
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")


CONFIDENCE_THRESHOLD = 0.12  

def get_response(user_input):
    """Preprocess the user input the same way as training data and predict a response.
    Falls back to a default reply if the model isn't confident enough."""
    processed_input = ' '.join(nltk.word_tokenize(user_input.lower()))
    probs = model.predict_proba([processed_input])[0]
    best_idx = probs.argmax()
    confidence = probs[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return "Sorry, I'm not sure I understand. Could you rephrase that?"

    return model.classes_[best_idx]


# Initializing Dash app
app = dash.Dash(__name__)
app.title = "Simple ML Chatbot"

app.layout = html.Div(
    style={
        'maxWidth': '600px',
        'margin': '40px auto',
        'fontFamily': 'Arial, sans-serif',
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'padding': '20px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
    },
    children=[
        html.H2("Chatbot", style={'textAlign': 'center'}),

        html.Div(
            id='chat-window',
            style={
                'height': '400px',
                'overflowY': 'auto',
                'border': '1px solid #eee',
                'borderRadius': '6px',
                'padding': '10px',
                'marginBottom': '15px',
                'backgroundColor': '#fafafa'
            },
            children=[]
        ),

        dcc.Store(id='chat-history', data=[]),

        html.Div(
            style={'display': 'flex', 'gap': '8px'},
            children=[
                dcc.Input(
                    id='user-input',
                    type='text',
                    placeholder='Type your message...',
                    style={'flex': '1', 'padding': '8px'},
                    n_submit=0
                ),
                html.Button('Send', id='send-button', n_clicks=0, style={'padding': '8px 16px'})
            ]
        )
    ]
)


@app.callback(
    Output('chat-window', 'children'),
    Output('chat-history', 'data'),
    Output('user-input', 'value'),
    Input('send-button', 'n_clicks'),
    Input('user-input', 'n_submit'),
    State('user-input', 'value'),
    State('chat-history', 'data'),
)
def update_chat(n_clicks, n_submit, user_message, history):
    if not user_message:
        return dash.no_update, dash.no_update, dash.no_update

    history = history or []

    bot_reply = get_response(user_message)

    history.append({'sender': 'user', 'text': user_message})
    history.append({'sender': 'bot', 'text': bot_reply})

    chat_bubbles = []
    for msg in history:
        is_user = msg['sender'] == 'user'
        chat_bubbles.append(
            html.Div(
                msg['text'],
                style={
                    'textAlign': 'right' if is_user else 'left',
                    'backgroundColor': '#DCF8C6' if is_user else '#F1F0F0',
                    'padding': '8px 12px',
                    'borderRadius': '12px',
                    'margin': '6px 0',
                    'display': 'inline-block',
                    'maxWidth': '80%',
                    'float': 'right' if is_user else 'left',
                    'clear': 'both'
                }
            )
        )

    return chat_bubbles, history, ''


if __name__ == '__main__':
    app.run(debug=True)