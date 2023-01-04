import streamlit as st
from transformers import BertForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import pipeline


st.set_page_config(
    page_title="QA - BERT Model!!!")


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([8, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("🔎 BERT QA App! ")
    st.header("")

context = (
           "The Information and Communications Technology Council (ICTC) "
           "A not-for-profit national centre of expertise for the digital economy, we are the trusted source for evidence-based policy advice, "
           "forward looking research, and creative capacity building programs, with a team of over 90 qualified professionals across Canada."
           "OUR MISSION: Strengthen Canada’s digital advantage in a global economy. "
           "OUR VISION:Foster globally competitive Canadian industries and a prosperous society empowered by innovative digital solution."
           "OUR VALUES: The Information and Communications Technology Council (ICTC) is committed to providing an atmosphere free "
           "from barriers that promotes equity and diversity. ICTC celebrates and welcomes the diversity of all employees, stakeholders, and external personnel. "
           "By valuing a diverse workforce, ICTC is committed to hiring practices that are fair and equitable. ICTC also supports a workplace environment "
           "and a corporate culture that is built on our TRUST values that encourage equal employment and career prospects for all employees."
           "ICTC EXECUTIVE TEAM : Our Leadership : Namir Anani is the President & CEO of ICTC. Alexandra Cutean is the Chief Research Officer and Rob Davidson is the Director of Data Science. "
           "annual report 2017-2018: https://www.ictc-ctic.ca/wp-content/uploads/2015/12/ICTC_AnnualReport_2017-18-Final-EN-2.pdf. "
           "annual report 2018-2019: https://www.ictc-ctic.ca/wp-content/uploads/2019/08/ICTC_ANNUAL-REPORT19_OFFICIAL.pdf. "
           "annual report 2019-2020: https://www.ictc-ctic.ca/wp-content/uploads/2020/11/Annual-Report_2020_EN-2.pdf. "
           "annual report 2020-2021: https://www.ictc-ctic.ca/wp-content/uploads/2021/08/Annual-Report-2020-2021.pdf. "
           "annual report 2021-2022: https://www.ictc-ctic.ca/wp-content/uploads/2022/10/ICTC-Annual-Report-2021-2022.pdf. "
           "Alexandra Cutean is the Chief Research Officer (CRO) at the Information and Communications Technology Council (ICTC). "
           "She leads research on transformative technologies, labour market and skill needs, and economic trends all with the ultimate goal "
           "of providing in-depth and timely research, analysis and policy considerations for the Canadian digital economy. "
           "Alexandra’s work experience extends across Canada, the United States, and Europe, with previous roles at PwC, KPMG, and the European Commission, among others. "
           "Alexandra holds two Master of Science (MSc) degrees from the University of Amsterdam – one in International Relations and Foreign Affairs, and another "
           "in Conflict Resolution and Negotiation. She also holds a post-graduate diploma in International Business from Humber College, a post-graduate certificate "
           "in Professional Communication from the University of British Columbia, and a BA in Political Science and English from Wilfrid Laurier University."
           "Rob Davidson is the Director of Data Science at the Information and Communications Technology Council (ICTC), an independent, non-profit think tank. "
           "Rob Davidson is a 25-year seasoned veteran of the software industry and has excelled in senior roles, ranging from Chief Technologist, "
           "Vice-President of Product Management to Director of Marketing & Communications. "
           "Rob has spoken at national and international events on emerging technologies, AI, and open data and government. "
           "Rob is a passionate open data advocate, promoting the use of open data for social good and business creation. "
           "He is a current member of and former co-chair of Canada’s Open Government Multi-Stakeholder Forum.  "
           "In June 2016, Rob founded the Open Data Institute Ottawa Node to help crystallize the open data movement in Ottawa."
           )

with st.expander("ICTC", expanded=True):

    st.write(
        context
    )

    st.markdown("")

st.markdown("")
st.markdown("Do you have a question about the executive team or ICTC in general ❓")
user_input = st.text_area('Type your question here.')
button = st.button("Ask")

@st.cache(allow_output_mutation=True)
def get_model():

    modelname = 'deepset/tinyroberta-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp


nlp = get_model()

if user_input and button :

    result = nlp({
    'question': user_input,
    'context': context
  })
    st.write("Answer: ",result['answer'])
