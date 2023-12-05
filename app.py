import streamlit as st
import pandas as pd
import random
from major_embeddings import EmbeddingsCalculator
from embeddings import allEmbeddingsCalculator
embeddings_calculator = allEmbeddingsCalculator()


def get_mcs(input_text):
    ta = EmbeddingsCalculator(model_name='bert-base-uncased')
    cs = ta.calculate_confidence_score(input_text)
    print(cs)
    return cs

def get_gcs(input_text):
    ta = allEmbeddingsCalculator()
    cs = ta.calculate_confidence_score(input_text)
    print(cs)
    return cs

def get_ms(input_text):
    ta = EmbeddingsCalculator(model_name='bert-base-uncased')
    major_scores = ta.calculate_weighted_average_similarity(input_text)
    print(major_scores)
    return major_scores

def get_gs(input_text):
    ta = allEmbeddingsCalculator()
    gs = ta.calculate_weighted_average_similarity(input_text)
    print(gs)
    return gs
def get_answer(input_text, input_major):
    ta = EmbeddingsCalculator(model_name='bert-base-uncased')

    assessment, top_majors_list = ta.evaluate_major(input_text, input_major)

    return assessment, top_majors_list

def map_to_original_major(formatted_major):
    term_replace = {
        'Accounting.and.finance': 'accounting.and.finance',
        'Agriculture Environment': 'agriculture_environment',
        'Archeology': 'archeology',
        'Architecture Art': 'architecture_art',
        'Biology': 'biology',
        'Chemical.engineering': 'chemical.engineering',
        'Communication Info': 'communication_info',
        'Development.studies': 'development.studies',
        'Economics': 'economics',
        'Electrical & Comp. Engineering': 'electrical & comp. engineering',
        'English.language.and.literature': 'english.language.and.literature',
        'History': 'history',
        'Linguistics': 'linguistics',
        'Mathematics Statistic': 'mathematics_statistic',
        'Mechanical.engineering': 'mechanical.engineering',
        'Medicine Health': 'medicine_health',
        'Philosophy': 'philosophy',
        'Physics': 'physics',
        'Political.science': 'political.science',
        'Psychology': 'psychology',
        'Sociology': 'sociology'
    }

    return term_replace.get(formatted_major, "Unknown Major")
def map_to_formatted_major(original_major):
    original_to_formatted = {
        'accounting.and.finance': 'Accounting.and.finance',
        'agriculture_environment': 'Agriculture Environment',
        'archeology': 'Archeology',
        'architecture_art': 'Architecture Art',
        'biology': 'Biology',
        'chemical.engineering': 'Chemical.engineering',
        'communication_info': 'Communication Info',
        'development.studies': 'Development.studies',
        'economics': 'Economics',
        'electrical & comp. engineering': 'Electrical & Comp. Engineering',
        'english.language.and.literature': 'English.language.and.literature',
        'history': 'History',
        'linguistics': 'Linguistics',
        'mathematics_statistic': 'Mathematics Statistic',
        'mechanical.engineering': 'Mechanical.engineering',
        'medicine_health': 'Medicine Health',
        'philosophy': 'Philosophy',
        'physics': 'Physics',
        'political.science': 'Political.science',
        'psychology': 'Psychology',
        'sociology': 'Sociology'
    }

    return original_to_formatted.get(original_major, "Unknown Major")

st.set_page_config(
    page_title="Reach Best LUR Bot", layout="centered", page_icon="logo.png", initial_sidebar_state="expanded"
)
st.write(
    '<div style="text-align: center;">'
    '<h1 style="color: #E1930F;">Reach Best LUR Bot</h1>'
    '</div>',
    unsafe_allow_html=True)
st.write(" ") 

st.write(
    """
    <style>

    h2 {
        text-align: center;
        margin-top: -24px; 
        font-size: 16px; 
        margin-bottom: -50px;
    }
    h3 {
        text-align: center;
        color: #80ACD4;
        margin-top: -20px; 
        font-size: 24px; 
    }
    h4 {
        text-align: center;
        margin-top: -20px; 
        font-size: 16px; 
    }
    h5 {
        font-size: 24px;
        margin-bottom: -20px;
    }

    h6{
        text-align:center;
        font-size: 24px;
        margin-top:-20px;
    }
    </style>
    <h3>Find Your ideal fit Universities based on your input</h3>
    """,
    unsafe_allow_html=True,
)
st.divider()

ID_weight = pd.read_csv('./ID weight_review.csv')


with st.sidebar:
    # with open("style.css") as f:  # change up the sidebar styling
    #    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.markdown(
           """
           <a href="https://app.reachbest.co/signup">
               <img src="https://i.imgur.com/CMfb6aI.png" style="max-width: 100%;">
           </a>
           """,
           unsafe_allow_html=True,
        )
    st.write(" ") 

  
    st.info("To access the official chat bot trained for over 1000+ Universities, check out [Reach Best](https://app.reachbest.co/signup)!", icon="🧠")
#     major_topics = [
#         "Academic Excellence and Opportunities",
#         "Social networks and Campus Atmosphere",
#         "Dynamics and individual experiences",
#         "Campus Beauty and Resources",
#         "Transfer Student Experience",
#         "Career and Major Focus"
#     ]


#     with st.expander("General Aspects:", expanded=False):
#         general_aspects = """
#             - <span class="copyable-text">Policy of Administration and Financial Aid</span>
#             - <span class="copyable-text" >Career and Academic Opportunities</span>
#             - <span class="copyable-text" >Academic Quality and online learning</span>
#             - <span class="copyable-text" >Admission</span>
#             - <span class="copyable-text">Diversity and Inclusion</span>
#             - <span class="copyable-text">Technology and Computer Labs</span>
#             """
#         st.markdown(general_aspects, unsafe_allow_html=True)


#     with st.expander("Major Specific Aspects:", expanded=False):
#         major_specific_aspects = "\n".join(f"- <span class='copyable-text'>{topic}</span>" for topic in major_topics)
#         st.markdown(major_specific_aspects, unsafe_allow_html=True)
    st.write(" ") 

    general_questions = [
        "What type of friends do you get along with and why?",
        "What’s your favorite season of the year and why? Do you prefer sunny, cloudy, rainy or snowy days?",
        "What do you like doing for fun? What type of places do you like hanging out at and why?",
        "What is the one thing you love the most about school and why?",
        "Do you prefer to dress casually or more formally and why?",
        "I want to answer multiple questions here",
        "I want to discuss other aspects of my ideal college life"
    ]
    major_questions = [
        "What do you call you favourite teacher,Mr or first name.Do you like it? Why?",
        "What's the way of teaching of your favourite teacher,lecture vs discussion, what's the characteristic you most appreciate? Why?", 
        "Do you prefer taking exams or writing research papers? Why?",
        "Would you rather be the only student in a super advanced class or be in a group class where you learn at the same rhythm with your classmates? Why?",
        "Which do you like, get higher scores for more attending and engaging vs or leting exams to dertermain your grade? Why?",
        "Describe the vibe in your favourite class, were you participating a lot, why?",
        "Which do you hate more, monthly long projects vs weekly homework, why?",
        "For the  project your most proud of,which parts do you enjoy,reading literature and find scientific solution? Why?",
        "I want to answer multiple questions here",
        "I want to discuss other aspects of my major specific preference"
    ]
    selected_general_question = st.sidebar.selectbox("Choose your first question", general_questions, key="select_general")
    st.write(" ") 

    selected_major_question = st.sidebar.selectbox("Choose your second question", major_questions, key="select_major")



col4, col5 = st.columns([5, 5])
with col4:
    username = st.text_input(
        label="Save your personalized model",
        placeholder="Enter your name"
    )

with col5:
    major_options = ['Accounting.and.finance',
                 'Agriculture Environment',
                 'Archeology',
                 'Architecture Art',
                 'Biology',
                 'Chemical.engineering',
                 'Communication Info',
                 'Development.studies',
                 'Economics',
                 'Electrical & Comp. Engineering',
                 'English.language.and.literature',
                 'History',
                 'Linguistics',
                 'Mathematics Statistic',
                 'Mechanical.engineering',
                 'Medicine Health',
                 'Philosophy',
                 'Physics',
                 'Political.science',
                 'Psychology',
                 'Sociology']

    sorted_major_options = sorted(major_options)

    major = st.selectbox(
        'Select Major',
        sorted_major_options)

st.write(" ") 


if 'random_question1' not in st.session_state:
    st.session_state['random_question1'] = random.choice(general_questions)
if 'random_question2' not in st.session_state:
    st.session_state['random_question2'] = random.choice(major_questions)

st.write(selected_general_question, unsafe_allow_html=True)

answer1 = st.text_area(selected_general_question, key="text_area1",placeholder='Write here',label_visibility='collapsed')
st.write(" ") 
st.write(selected_major_question, unsafe_allow_html=True)

answer2 = st.text_area(selected_major_question, key="text_area2",placeholder='Write here',label_visibility='collapsed')
st.write(" ") 

if username:
    if username in ID_weight["name"].tolist():
        user_data = ID_weight[ID_weight["name"] == username].iloc[-1]

        user_last_answers = pd.DataFrame({
            'Category': ['General College Life', 'Major Specific Life'],
            'Your Last Answer': [user_data["lastphrase1"], user_data["lastphrase2"]]
        })

#         st.write(user_last_answers.to_html(index=False), unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

        weight_flag= float(user_data["Model1weight"]) if 'Model1weight' in user_data and isinstance(user_data["Model1weight"], (int, float)) else 0.5
    else:
        weight_flag = 0.5
        st.write("No previous data found for user: {}".format(username))

    col1, col2, col3 = st.columns([2, 6, 2])
    with col1:
        st.write("<p style='text-align: center'>Major specific</p>", unsafe_allow_html=True)
    with col2:
        weight = st.slider('Adjust your preference', 0.0, 1.0, weight_flag, label_visibility='collapsed')
    with col3:
        st.write("<p style='text-align: center'>General review</p>", unsafe_allow_html=True)
if 'lists' not in st.session_state:
    st.session_state.lists =0          
if 'unianswer' not in st.session_state:
    st.session_state.unianswer =0  
        
if 'cs1' not in st.session_state:
    st.session_state.cs1 =0  
if 'weight' not in st.session_state:
    st.session_state.weight = 0      
if 'cs2' not in st.session_state:
    st.session_state.cs2 = 0       
if 'selected_university' not in st.session_state:
    st.session_state.selected_university = None
if 'test' not in st.session_state:
    st.session_state.test = 0   
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_major' not in st.session_state:
    st.session_state.df_major = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if st.button("Recommend"):
    if not username:
        st.warning("Please enter your name.")
    elif not answer1 or not answer2:
        st.warning("Please provide answers for both text inputs.")
    else:
        pd.concat([ID_weight, pd.DataFrame(
            [username, weight, 1 - weight,
             answer1, answer2], index=ID_weight.columns).T],
                  ignore_index=True).to_csv(
            './ID weight_review.csv', index=False)
        with st.spinner('Running...'):

            df = get_gs(input_text=answer1)
#             df = df.reset_index()
            df.columns = ['University', 'WAS', 'Highest_Prob_Topic', 'Most_Relevant_Review','Most_Relevant_Author','Most_Relevant_Created']

            df_major = get_ms(input_text=answer1)
#             df_major = df_major.reset_index()
            df_major.columns = ['University', 'WAS', 'highest_prob_major', 'Most_Relevant_Review','Most_Relevant_faculty','Most_Relevant_course','Most_Relevant_Created']

            df_copy = df.set_index('University')
            df_was2 = df_copy[['WAS']]
            df_was2.columns = ['WAS2']
            df_major_copy = df_major.set_index('University')
            df_was1 = df_major_copy[['WAS']]
            df_was1.columns = ['WAS1']

            df_final = pd.concat([df_was1,df_was2],axis=1).reset_index()
            df_final['WAS'] = df_final['WAS1']*weight+df_final['WAS2']*(1-weight)
            df_final = df_final.sort_values(by='WAS', ascending=False)[:7].reset_index().reset_index()

            df_final["index"] = df_final["level_0"]+1
            df_final = df_final[['index','University','WAS1','WAS2','WAS']]
            df_final.columns = ["Rank", "University",  "Major Review Prob.", "General Review Prob.","Weighted Avg."]

#             st.success("✅Recommend success")
            cs1 = get_mcs(answer1)
            cs2 = get_gcs(answer2)
            unianswer, lists = get_answer(answer2, map_to_original_major(major))
            lists = [map_to_formatted_major(major) for major in lists]

            if df_final is not None:
                st.session_state.df_final = df_final
                st.session_state.test += 1
            if df is not None:
                st.session_state.df = df
            if df_major is not None:
                st.session_state.df_major = df_major
            if cs1 is not None:
                st.session_state.cs1 = cs1
                
            if cs2 is not None:
                st.session_state.cs2 = cs2
            if weight is not None:
                st.session_state.weight = weight
            if unianswer is not None:
                st.session_state.unianswer = unianswer
            if lists is not None:
                st.session_state.lists = lists                
if st.session_state.test >= 1:
    

    st.write(
        """
        <style>
        h3 {
            text-align: center;
            color: #80ACD4;
            margin-top: -20px; 
            font-size: 24px; 
        }

        </style>
        <h3>Here are your personalized results</h3>
        """,
        unsafe_allow_html=True,
    )    
#     st.header("Here are your personalized results")
    st.divider()
    if st.session_state.unianswer == 'perfect':
        advice_message = "🤩 <span style='font-size: 18px; font-weight: bold;'>The major you choose could be a perfect fit for you. Here are our pieces of advices:</span>"
        st.markdown(advice_message, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        advice_list = lists[:3] 
        with col1:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[0]}</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[1]}</span></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[2]}</span></div>", unsafe_allow_html=True)        

    elif st.session_state.unianswer == 'good':
        advice_message = "😊 <span style='font-size: 18px; font-weight: bold;'>The major you choose could be a good fit for you. Here are our pieces of advices:</span>"
        st.markdown(advice_message, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        advice_list = lists[:3] 
        with col1:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[0]}</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[1]}</span></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[2]}</span></div>", unsafe_allow_html=True)        
      

    elif st.session_state.unianswer == 'reasonable': 
        advice_message = "😬 <span style='font-size: 18px; font-weight: bold;'>The major you choose could be a reasonable fit for you. Here are our pieces of advices:</span>"
        st.markdown(advice_message, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        advice_list = lists[:3] 
        with col1:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[0]}</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[1]}</span></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[2]}</span></div>", unsafe_allow_html=True)        
 
    elif st.session_state.unianswer == 'bad':
    
        advice_message = "😢 <span style='font-size: 18px; font-weight: bold;'>The major you choose could be a bad fit for you. Here are our pieces of advice for you:</span>"
        st.markdown(advice_message, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        advice_list = lists[:3] 
        with col1:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[0]}</span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[1]}</span></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align: center;'>"
                "<span style='color: orange; font-size: 20px; font-weight: bold;'>"
                f"{advice_list[2]}</span></div>", unsafe_allow_html=True)        
 
    st.write(" ") 
    st.write(" ") 

    advice_school = "🏫 <span style='font-size: 18px; font-weight: bold;'> Here are our recommended schools just for you:</span>"
    st.markdown(advice_school, unsafe_allow_html=True)
    st.write(" ") 

    st.dataframe(st.session_state.df_final, use_container_width=True, hide_index=True)
    st.write(" ") 
    st.info(f'How confident is the model? \n\nBased on your personal weight, our model accuracy is  **{100*round((0.41*(1-st.session_state.weight)+0.48*st.session_state.weight), 2)}%** ',icon="ℹ")
    st.write(" ") 
    st.info(
    f'The relevance of your first answer to our Niche reviews is **{round(st.session_state.cs2*100, 2)}%**\n\nThe relevance of your second answer to our ratemyprofessor reviews is **{round(st.session_state.cs1*100, 2)}%**',
    icon="ℹ")
    st.write(" ") 
    default_option = "Select a University to view reviews"
    university_options = [default_option] + list(st.session_state.df_final["University"].unique())
    advice_Review = "🔍 <span style='font-size: 18px; font-weight: bold;'> Here are some reviews that maybe relevant to your answer:</span>"
    st.markdown(advice_Review, unsafe_allow_html=True)
    selected_university = st.selectbox("", university_options)
    st.session_state.selected_university=selected_university
    if selected_university and selected_university != default_option:
        st.session_state.selected_university = selected_university
        filtered_df = st.session_state.df[st.session_state.df['University'] == selected_university]

        if not filtered_df.empty:
            general_review = filtered_df['Most_Relevant_Review'].values[0]
            general_topic = filtered_df['Highest_Prob_Topic'].values[0]
            general_author = filtered_df['Most_Relevant_Author'].values[0]
            general_date = filtered_df['Most_Relevant_Created'].values[0]

            # Similar check for major reviews
            filtered_df_major = st.session_state.df_major[st.session_state.df_major['University'] == selected_university]
            if not filtered_df_major.empty:
                major_review = filtered_df_major['Most_Relevant_Review'].values[0]
                major = filtered_df_major['highest_prob_major'].values[0]
                professor = filtered_df_major['Most_Relevant_faculty'].values[0]
                course = filtered_df_major['Most_Relevant_course'].values[0]
                major_date = filtered_df_major['Most_Relevant_Created'].values[0]

                st.write("🎓" + general_topic)
                st.write("- \"" + general_review + "\"\n")
                st.write("‎ ‎‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎ ‎ ‎ ‎ ‎  ‎‎   ‎‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎ ‎ ‎ ‎ ‎  ‎‎   - *"  + general_author + " on " + general_date + "*\n")
                st.write("")
                st.write("📚" + major)
                st.write("- \"" + major_review + "\"\n")
                st.write("‎ ‎‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎ ‎ ‎ ‎ ‎  ‎‎   ‎‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎ ‎ ‎ ‎ ‎  ‎‎   - *"  + course + " with " + professor + " on " + major_date + "*\n")
        else:
            st.write(" ") 
    st.write(" ") 
    st.write(" ") 
    st.write(" ") 
    st.write(" ") 
    st.write(" ")
    st.write(" ") 
    st.write(" ") 

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        double_up = st.button("👍👍", use_container_width=True)

    with col2:
        up = st.button("👍", use_container_width=True)

    with col3:
        down = st.button("👎", use_container_width=True)

    with col4:
        double_down = st.button("👎👎", use_container_width=True)

    feedback = None

    if double_up:
        feedback = "Really good"

    if up:
        feedback = "Good"

    if down:
        feedback = "Bad"

    if double_down:
        feedback = "Really bad"

    if feedback is not None:
                st.success("Thanks for your feedback")
                st.session_state.generated = True




st.markdown(
    "<p style='text-align:center; color: #C5C5C5;'>Free prototype preview. AI may sometimes provide innacurate "
    "information. This model was trained on Nich reviews and Rate My Professors Reviews. "
    "</p>",
    unsafe_allow_html=True,
)
