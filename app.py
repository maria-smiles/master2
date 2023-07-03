import streamlit as st
import pandas as pd
import numpy as np
import os


st.title('Language grade')
st.markdown('Here you can find out if the movie suits your level of English.')


sideb = st.sidebar
sideb.markdown('1. Download files here')

uploaded_files = sideb.file_uploader("Choose SRT file(s). Several are possible. ", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        #bytes_data = uploaded_file.read()
        #.write("filename:", uploaded_file.name)
        with open(os.path.join("Dir",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())         
    
sideb.markdown('2. Push button below when files will be downloaded')
submit = sideb.button("button below")   
path = 'Dir' # Путь к папке, в которую будет загружаться клиентский файл субтитров  

    
def preproc_subs():

    # Подготовка входящего субтитра к векторизации
    import pysrt
    import warnings
    import pandas as pd
    import nltk
    import re
    import os

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop = stopwords.words('english') + ["i'm"]

    del_n = re.compile('\n')                  # перенос каретки
    del_tags = re.compile('<[^>]*>')          # html-теги
    del_brackets = re.compile('\([^)]*\)')    # содержимое круглых скобок
    clean_text = re.compile('[^а-яa-z\s\']')  # все небуквенные символы кроме пробелов и апострофов
    del_spaces = re.compile('\s{2,}')         

    # Удаление символов
    def prepare_text(text):
        text = del_n.sub(' ', str(text).lower())
        text = del_tags.sub('', text)
        text = del_brackets.sub('', text)
        res_text = clean_text.sub('', text)
        res_text = res_text.lower()  #перевод в нижний регистр
        return del_spaces.sub(' ',res_text)
        
    # Удаление стоп-слов с вызовом функции с удалением символов
    def df_stop_words(Sr, stop):
        for itm in range(len(Sr)):
            line = Sr[itm]
            line = prepare_text(line)
            line = [w for w in line.split() if w not in stop]
            Sr[itm] = line
        return Sr    


    # Лемматизация, поиск корней с вызовом функции удаления стоп-слов   
    def lemmatise_srt(Sr, stop):
        df_stop_words(Sr, stop)
        stemmer = WordNetLemmatizer()
        for itm in range(len(Sr)):
            line = Sr[itm]
            line = [stemmer.lemmatize(w) for w in line]
            Sr[itm] =  str(line)
        return Sr
          

    

    df = pd.DataFrame(columns = ['Movie', 'Srt'])
    # Данные на вход - файл субтитров
    for file in os.listdir(path):
        infile = pysrt.open(os.path.join(path, file), encoding='iso-8859-1') # пользуемся этим при указании пути в папку сфайлом, а что с одиночным файлом?
        
        txt = infile.text # чтение субтитра
        df.loc[len(df)] = [file[:-4], txt]


    df['Srt'] = lemmatise_srt(df['Srt'], stop) 

    # Векторизация
    import pickle
    with open('pickle_dict.pkl', 'rb') as file: 
        eng_vocabulary = pickle.load(file)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    
    count = CountVectorizer(vocabulary=eng_vocabulary)
    bag = count.fit_transform(df['Srt'])
    tfidfconverter = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

    bag = tfidfconverter.fit_transform(bag).toarray() # преобразует текст в цифры

    with open('pickle_model.pkl', 'rb') as file: 
        model = pickle.load(file)
        
    # Предсказание уровня языка по субтитрам фильма

    predicted = model.predict(bag)
    df['Level_pred'] = pd.Series(predicted)
    df = df.drop(columns=['Srt'])
    df.to_csv('answer.csv', index=False)
    return df


if submit:
    df = preproc_subs()
    st.markdown('3. Read the answer')
    
    st.dataframe(data=df)
    


