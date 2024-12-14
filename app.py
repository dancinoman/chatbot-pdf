
import streamlit as st
import model_run as mr

def build_main_page():
    """
    Streamlit app entry point.
    """

    # Top of the page
    st.title("Document Question Answering")
    st.write('### Instructions')
    st.write("""
                1. Upload your pdf file
                2. Question AI about the Document
                3. When ready to download results, close the document
             """)
    # Get text from uploaded PDF

    # Bottom of the page
    uploaded_file = st.file_uploader("Upload Contract PDF", type="pdf")

    # Manage the reset state for new uploaded PDF
    if "chat_history"in st.session_state:
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

        if user_input := st.chat_input('Ask your question'):
            user_input = user_input.strip()

            if document_text:
                answer = mr.process_and_answer(user_input, document_text)

                st.session_state.chat_history.append({'message': f'**You**: {user_input}'})
                st.session_state.chat_history.append({'message':f"**AI**:'{answer}'"})
                for message in st.session_state['chat_history']:
                    st.markdown(message['message'])


if __name__ == "__main__":
    build_main_page()
