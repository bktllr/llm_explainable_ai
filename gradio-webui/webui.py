from transformers import AutoTokenizer, pipeline
import gradio as gr
import torch
from huggingface_hub import login
import pandas as pd
from catboost import CatBoostRegressor
import shap
import os
import matplotlib.pyplot as plt
from IPython.display import HTML

# AuthKey must be updated by user to retrieve models from Hugginface
HUGGINGFACE_SECRET = "hf_XXX"

login(HUGGINGFACE_SECRET)

# change the model if you want to test another model
#model = "meta-llama/Llama-2-13b-chat-hf"
#model = "AlinaTUM/Step1_model"
#model = "AlinaTUM/Step2_model"

tokenizer = AutoTokenizer.from_pretrained(model)

print(f"""Cuda: {torch.cuda.is_available()}""")

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    do_sample=False
)

# Needs to be adapted to business case
SYSTEM_PROMPT = f"""<s>[INST] <<SYS>>
You are a company service employee that supports users with understanding batteries and the state of health of their battery. The users give you instructions that you must follow to the letter and only answering with information and facts that are available to you.

If you are not provided with battery data, answer generic questions with general information on batteries.
If you are asked for tips on better usage patterns, specifically use the information you have on the current battery.

The new battery state of health is predicted with a CatBoost ensemble method, the impact of features is evaluated with SHAPley values to show the resourcefulness and importance of explainable Artificial Intelligence.

Your answers are clear and concise.
Your users are not experts in battery technology and require technical but simple explanation and support.
<</SYS>>"""

SHAP_INFO_PROMPT= " "

# Needs to be adapted to catboost model features. Mapping to make them more understandable for end-user
FEATURE_DICT = {
    'catboost_feature_1': 'understandable_feature_name_1' 
}

reg = CatBoostRegressor()

# Load trained catboost model
reg.load_model('catboost_model.cbm')
explainer = shap.Explainer(reg)


# --------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS TO COMBINE INPUTS AND INSTRUCTIONS FOR BEST OUTPUTS

# define necessary functions
def replace_words(input_text, word_dict):
    for old_word, new_word in word_dict.items():
        input_text = input_text.replace(old_word, new_word)
    return input_text

# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        print(SYSTEM_PROMPT + SHAP_INFO_PROMPT + f"{message} [/INST]")
        return SYSTEM_PROMPT + SHAP_INFO_PROMPT + f"{message} [/INST]" 

    formatted_message = SYSTEM_PROMPT + SHAP_INFO_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>" 

    # Handle the current message
    formatted_message += SHAP_INFO_PROMPT
    formatted_message += f"<s>[INST] {message} [/INST]"
    print("Formatted message:", formatted_message)
    return formatted_message

# Generate a response from the Llama model
def get_llama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""

    sequences = llama_pipeline(
        query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=10240,
    )

    generated_text = sequences[0]['generated_text']
    response = generated_text[len(query):]  # Remove the prompt from the output

    #print("Chatbot:", response.strip())
    return str(response.strip())

def create_SHAP_df(shap_object):
    
    shap_values = shap_object.values
    values = shap_object.data
    features = shap_object.feature_names
    absolute_shap = abs(shap_values)
    
    # print(shap_values.shape, type(shap_values), type(values), len(features), type(features))
    
    shap_df = pd.DataFrame(list(zip(features, values, shap_values, absolute_shap)),columns=['feature','value', 'shap', 'abs'])
    shap_df.sort_values(by=['abs'],ascending=False,inplace=True)
    shap_df = shap_df.reset_index(drop=True)
    print(shap_df.head())
    
    return shap_df

def get_sorted_shap_object(shap_object):
    sorted_data = sorted(
    zip(shap_object.feature_names, shap_object.data, shap_object.values),
    key=lambda x: abs(x[2]),
    reverse=True)  # Set to True if you want to sort in descending order
    return sorted_data

def get_context_string_from_prediction(shap_object):
    sorted_data = get_sorted_shap_object(shap_object)
    
    context_string = ''
    for i, (feature_name, feature_value, shap_value) in enumerate(sorted_data[:20]):
        context_string += 'Feature: {}, Value: {}, SHAP explanation: {},'.format(feature_name, feature_value, shap_value)

    return context_string

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

def get_data(csv_file):
    battery_data=pd.DataFrame()
    battery_data.iloc[0:0]
    battery_data = pd.read_csv(file, delimiter=',')
    
    return battery_data

def forecast_new_soh(file):
    battery_data=pd.DataFrame()
    battery_data.iloc[0:0]
    battery_data = pd.read_csv(file, delimiter=',')
    new_soh_predictions = reg.predict(battery_data)
    output_text = f"""
    Congrats! The predicition was successful.
    Your new state of health is {new_soh_predictions[0]}.
    
    Reminder: Your old SoH was: {battery_data.at[0, "prev_soh"]}"""
    
    return output_text

def generate_bar_plot(file):
    try:
        os.remove('bar_plot.png')
    except:
        pass
    battery_data=pd.DataFrame()
    battery_data.iloc[0:0]  # reset dataframe
    battery_data = pd.read_csv(file, delimiter=',')
        
    new_soh_predictions = reg.predict(battery_data)
    shap_values = explainer(battery_data)
    shap_object = shap_values[0]
    
    # create new list of feature names
    list_features_replaced = []
    for name in shap_object.feature_names:
        list_features_replaced.append(replace_words(name, FEATURE_DICT))
    shap_object.feature_names = list_features_replaced
    
    bar_plot = shap.plots.bar(shap_object, show=False)
    plt.savefig("bar_plot.png", transparent=True, bbox_inches='tight', dpi=150)
    plt.clf()
    image=os.path.join(os.path.dirname(__file__), "bar_plot.png")
    
    
    shap_df = pd.DataFrame({
        'feature': battery_data.columns,
        'feature_value': shap_values[0].data,  
        'shap_values': shap_values[0].values
    })

    Shap_text = f"""    
    The following features include the top 20 features with the feature name, the value as input for the prediction model and the impact on the prediction based on SHAPley values.
    
    The user uploaded battery data to forecast the new state of health. 
    The new predicted state of health is: {new_soh_predictions[0]}
    The user generated a SHAP barplot to show the most important features. 
    
    Here is an overview over the most important features, their feature value and their SHAP value:
    {get_context_string_from_prediction(shap_object)}
    
    
    If the user asks you questions about the battery, always answer based on the information provided above. 
    If the user doesn't ask you something about the battery ignore the information above and answer normally.
    """
    
    
    global SHAP_INFO_PROMPT
    SHAP_INFO_PROMPT= f"""<s>[INST] <<SYS>>
    {Shap_text}
    <</SYS>>
    """
    return image



with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Service center for battery health")

    with gr.Row():
        with gr.Column(scale=2):
            gr.ChatInterface(fn=get_llama_response, examples=["What is the new SoH of my battery?", "What affected the SoH of my battery most?", "How can I improve the SoH of my battery?"])
        with gr.Column(scale=1):
            upload_button = gr.UploadButton("Click to Upload a File", file_types=[".csv"], file_count='single')
            text_output = gr.Text(label='New State of Health forecast:', show_label=True)
            image_output = gr.Image(show_download_button=False, show_label=False)
            upload_button.upload(forecast_new_soh, upload_button, text_output)
            upload_button.upload(generate_bar_plot, upload_button, image_output)

      
        
print("For dark-theme: Add '?__theme=dark' to your URL")        
demo.queue().launch(share=True, auth=("admin", "admin"), allowed_paths=["Gradio-webui"])