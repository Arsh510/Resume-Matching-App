import streamlit as st
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Resume Matching App")
st.text("Aim of this project is to whether a candidate is qualified for a role based his \n or her education, experience, and other information captured on their resume.")

uploadedJD = st.file_uploader("Upload Job Description", type="pdf")

uploadedResume = st.file_uploader("Upload resume", type="pdf")

click = st.button("Process")


try:
    global job_description
    with pdfplumber.open(uploadedJD) as pdf:
        pages = pdf.pages[0]
        job_description = pages.extract_text()

except:
    st.write("Please Drop the file...")        


try:
    global resume
    with pdfplumber.open(uploadedResume) as pdf:
        pages = pdf.pages[0]
        resume = pages.extract_text()

except:
    st.write("Please Drop the file...")

# logic
def getResult(JD_txt, resume_txt):
    ''' JDFileObj = open(job_description,'rb')
     pdfReader = PyPDF2.PdfFileReader(JDFileObj)
     pageObj = pdfReader.getPage(0)
     JD_txt = pageObj.extractText()
     JDFileObj.close()


     ResumeFileObj = open(resume,'rb')
     pdfReader = PyPDF2.PdfFileReader(ResumeFileObj)
     pageObj = pdfReader.getPage(0)
     resume_txt = pageObj.extractText()
     ResumeFileObj.close()'''

    content = [JD_txt, resume_txt]

    cv = CountVectorizer()

    matrix = cv.fit_transform(content)

    similarity_matrix = cosine_similarity(matrix)

    match = similarity_matrix[0][1] * 100

    return match


# button

if click:

    st.write(getResult(job_description, resume))
