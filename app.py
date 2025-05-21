import streamlit as st
from qa_chain import main_chain

st.set_page_config(page_title="المساعد الضريبي", page_icon="🧾", layout="centered")

st.title("🧾 المساعد الضريبي الذكي")
st.markdown("اسأل أي سؤال متعلق بالضريبة المصرية وسنجيبك بناءً على النصوص القانونية الرسمية فقط.")

# User input
question = st.text_input("✍️ اكتب سؤالك هنا:")

# Button to submit
if st.button("أجبني"):
    if question.strip() == "":
        st.warning("من فضلك اكتب سؤالًا أولًا.")
    else:
        with st.spinner("جارٍ البحث والإجابة..."):
            try:
                answer = main_chain.invoke(question)
                if answer.strip() == "":
                    st.info("لا أعلم")
                else:
                    st.success("✅ الإجابة:")
                    st.markdown(answer)
            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة: {e}")
