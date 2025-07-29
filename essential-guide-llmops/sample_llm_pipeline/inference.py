import os
import streamlit as st
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

# تغییر مسیر به دایرکتوری مدل
os.chdir("/LLaMA-Factory")

# بارگذاری مدل با کش
@st.cache_resource
def load_model():
    args = dict(
        model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit",
        # adapter_name_or_path="llama3_lora",
        adapter_name_or_path="llama3_lora_identity",
        # adapter_name_or_path="llama3_lora_identity_final",
        template="llama3",
        finetuning_type="lora",
    )
    return ChatModel(args)

chat_model = load_model()

# تنظیمات صفحه
st.set_page_config(page_title="چت با LLaMA 3", layout="wide")

# اضافه کردن استایل راست‌چین
st.markdown("""
    <style>
        body, .stApp {
            direction: rtl;
            text-align: right;
        }
        .stTextInput input {
            direction: rtl;
            text-align: right;
        }
        .stChatMessage {
            direction: rtl;
            text-align: right;
        }
        .css-1cpxqw2, .css-12oz5g7 {
            direction: rtl;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

# عنوان صفحه
st.title("🦙 چت با مدل LLaMA 3")

# مقداردهی اولیه پیام‌ها
if "messages" not in st.session_state:
    st.session_state.messages = []

# نوار کناری
with st.sidebar:
    st.header("🛠 تنظیمات")
    if st.button("🗑 پاک‌کردن تاریخچه"):
        st.session_state.messages = []
        torch_gc()
        st.success("تاریخچه پاک شد.")

# ورودی کاربر
user_input = st.chat_input("پیام خود را بنویسید...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # نمایش پیام کاربر
    with st.chat_message("user"):
        st.markdown(user_input)

    # پاسخ مدل
    with st.chat_message("assistant"):
        response = ""
        response_placeholder = st.empty()
        for chunk in chat_model.stream_chat(st.session_state.messages):
            response += chunk
            response_placeholder.markdown(response + "▌")
        response_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# پاک‌سازی
torch_gc()
