import os,pickle,re

def zaladuj(plik,fn):
    if os.path.exists(plik):
        with open(plik, 'rb') as f:
            print('z cache')
            return pickle.load(f)
    text=fn()
    print('od nowa')
    with open(plik, 'wb') as f:
        pickle.dump(text, f)
    return text

def clean_text(text):
    text=text.lower()
    # Remove 'subject:' from the start of each record
    
    text = re.sub('re :', '', text)
    # Remove everything from '-------forwarded by' to 'subject:', where number of '-' can vary
    text = re.sub('- - - -.*forwarded by.*?(?=subject)', '', text,flags=re.DOTALL)
    text = re.sub('enron on.* - - -.*?(?=to :)', '', text,flags=re.DOTALL)
    text = re.sub('sender :.*?(?=subject :)', '', text,flags=re.DOTALL)
    
    # Remove everything from 'to :' to 'cc :'
    text = re.sub('to :.*?(?=cc :)', '', text,flags=re.DOTALL)
    
    # Remove everything from 'cc :' to 'subject:'
    text = re.sub('cc :.*?(?=subject :)', '', text,flags=re.DOTALL)
    text = re.sub('from :.*?(?=subject :)', '', text,flags=re.DOTALL)
    text = re.sub('(subject :|subject:)', '', text)
    return text