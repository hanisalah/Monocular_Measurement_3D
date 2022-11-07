import streamlit as st
from pathlib import Path

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


st.title('3D Object Identification and Measurement')

md_install = read_markdown_file('docs/install.md')
md_gen_dset = read_markdown_file('docs/generate_dataset.md')
md_run = read_markdown_file('docs/run.md')

st.markdown(md_install, unsafe_allow_html=True)
st.markdown(md_gen_dset, unsafe_allow_html=True)
st.markdown(md_run, unsafe_allow_html=True)
