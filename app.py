
import streamlit as st
import model_run as mr
import time

def build_main_page():
    """
    Streamlit app entry point.
    """

    def use_chat_box(user_reply_and_wait):

        if user_reply_and_wait:
            for i, message in enumerate(st.session_state['chat_history']):
                if i != len(st.session_state['chat_history']):
                    st.markdown(message['message'])

        else:
            st.markdown(st.session_state['chat_history'][-1]['message'])

    # Top of the page
    st.title("Doc Explainer")
    st.write('### Instructions')
    st.write("""
                1. Upload your pdf file
                2. Question AI about the Document
                3. When ready to download results, close the document
             """)

    # Bottom of the page
    uploaded_file = st.file_uploader("Upload Contract PDF", type="pdf")

    # Manage the reset state for new uploaded PDF
    if "chat_history" in st.session_state:
        if st.session_state["chat_history"] != [] and uploaded_file == None:

            # Convert message structure to a text file
            message_history = []
            for item in st.session_state['chat_history']:
                message_history.append(item['message'])
            binary_data = '\n\n'.join(message_history)

            st.download_button('document prompt results.txt', binary_data, 'document prompt results.txt')

     # Initialize chat history
    if "chat_history" not in st.session_state or uploaded_file == None:
        st.session_state["chat_history"] = []


    # Uploaded file is set and ready to parse
    if uploaded_file:
        document_text = mr.get_from_pdf(uploaded_file)

        # If there is document valid, user input is available
        if document_text:

            # User input and chatbox mangament
            if user_input := st.chat_input('Ask your question'):
                user_input = user_input.strip()
                st.session_state.chat_history.append({'message': f'**You**: {user_input}'})

                use_chat_box(True)

                with st.spinner('Searching'):
                    answer = mr.process_and_answer(user_input, document_text)

                st.session_state.chat_history.append({'message':f"**AI**:'{answer}'"})
                use_chat_box(False)

if __name__ == "__main__":
    build_main_page()
