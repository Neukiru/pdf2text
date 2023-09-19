import streamlit as st
import streamlit.components.v1 as sc
import requests
import json
from urllib import parse
  
ss = st.session_state


def main():

    title = 'Pdf to text'
    st.set_page_config(page_title=title, page_icon="", layout="centered")
    if 'uploaded_file_list' not in ss:

        ss.uploaded_file_list = None
        ss.payload = None
        ss.byte_array = None

    sc.html("""""",height=190)
    with st.form("my-form", clear_on_submit=True):
        st.selectbox('Select',options=['option 1','option 2','option 3'])
        ss.uploaded_file_list = st.file_uploader('Select file', type="pdf",accept_multiple_files=True)
        submitted = st.form_submit_button("Submit")
        

        response = None
        process_url = 'http://backend:8001/process/' # for docker
        # process_url = 'http://host.docker.internal:8001/process/' # for docker
        # process_url = 'http://localhost:8001/process/' # for testing outisde of docker
        
        if submitted:
            ss.byte_array = []
            pdf_names = []
            for file in ss.uploaded_file_list:
                ss.byte_array.append(bytes(file.getvalue()))
                pdf_names.append(file.name)

            if pdf_names:
                with st.spinner('Uploading and processing...'):
                    for index,value in enumerate(pdf_names):
                        response = requests.post(url=process_url, params = {"pdf_name":value}, files={'batch':ss.byte_array[index]})                   
                        if response.status_code != 200:
                            st.error('Error. Please upload files and try again')
                        
                        if response and response.status_code == 200:
                            text = json.loads(response.text)
                            for key in text["pdf"]:
                                url = parse.quote(text["pdf"][key])
                                st.markdown(f'download <a href="data:text/plain;charset=utf-11,{url}" download="{key}.txt">{key}.txt</a>', unsafe_allow_html=True)

                    ss.payload = None
            else:
                st.error('please select pdf files first and then submit')
                    

if __name__ == '__main__':
    main()

